"""A minimal Transformer (Vaswani et al., 2017) in one file: encoder-decoder, runs on a CPU.

    python transformer_minimal.py             # self-tests, then trains a small model on a copy task (about a minute)
    python transformer_minimal.py --steps 600 # more training steps

The code is adapted from The Annotated Transformer (Harvard NLP, https://github.com/harvardnlp/annotated-transformer,
MIT licence, Copyright (c) 2018 Alexander Rush; see NOTICE) and follows its structure, with these differences:
  * the sub-layer wrapper is Pre-LN (x + Dropout(Sublayer(LayerNorm(x)))); a Post-LN version, the one in the paper,
    is included for comparison;
  * the source embedding, the target embedding and the output layer can share one weight matrix (paper, section 3.4)
    when both languages use one vocabulary (tie_weights=True);
  * masks are built by one helper, make_masks, from the *decoder input* (the target shifted right by one);
  * the loss is cross-entropy with label smoothing, written out in a few lines.

Self-tests check the properties the blog post states: shapes, that the causal mask lets no information flow from
future positions, that padding does not change the result, the parameter count of the paper's base model, and that
setting masked scores to -inf and multiplying them by 0 behave differently.
"""
import argparse
import copy
import math
import time

import torch
import torch.nn as nn

PAD, BOS, EOS = 0, 1, 2   # special token ids used by the toy task


# ----------------------------------------------------------------------------------------------------------------
# Input: token embedding and sinusoidal position encoding
# ----------------------------------------------------------------------------------------------------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)   # lookup table, vocab x d_model
        self.d_model = d_model

    def forward(self, x):                          # x: token ids, (batch, n)
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)                       # pos
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 1 / 10000^(2i/d)
        pe[:, 0::2] = torch.sin(position * div_term)                           # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)                           # odd dimensions
        self.register_buffer("pe", pe.unsqueeze(0))                            # not trained

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# ----------------------------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------------------------
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)    # score, then scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)           # masked positions get a very small score
    p_attn = scores.softmax(dim=-1)                            # softmax over each row
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn @ value, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k, self.h = d_model // h, h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])   # W^Q, W^K, W^V, W^O
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)                           # the same mask for every head
        B = query.size(0)
        # 1) one projection, then split into h heads: (B, n, d_model) -> (B, h, n, d_k)
        query, key, value = [
            lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 2) all heads attend in parallel
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) concatenate: (B, h, n, d_k) -> (B, n, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        # 4) multiply by W^O
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# ----------------------------------------------------------------------------------------------------------------
# Residual connection and layer normalisation: Post-LN (the paper) and Pre-LN (used below)
# ----------------------------------------------------------------------------------------------------------------
class SublayerConnectionPostLN(nn.Module):
    """The paper: LayerNorm(x + Dropout(Sublayer(x)))."""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class SublayerConnection(nn.Module):
    """Pre-LN: x + Dropout(Sublayer(LayerNorm(x))). The encoder and the decoder then end with one more LayerNorm."""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ----------------------------------------------------------------------------------------------------------------
# Encoder and decoder
# ----------------------------------------------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn, self.feed_forward = self_attn, feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))      # Q = K = V = x
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)                                                    # this is the memory


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn, self.src_attn, self.feed_forward = self_attn, src_attn, feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   # causal self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))    # Q = x, K = V = memory
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)                           # every layer reads the same memory
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)                                  # d_model -> vocabulary size

    def forward(self, x):
        return self.proj(x).log_softmax(dim=-1)                                # log-probabilities


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed                  # token embedding + position encoding
        self.generator = generator                                             # Linear + log-softmax

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)                     # memory

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, tie_weights=False):
    """The paper's base model with the defaults. tie_weights=True shares one matrix between the two embeddings and
    the output layer (paper, section 3.4); it needs src_vocab == tgt_vocab."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    if tie_weights:
        assert src_vocab == tgt_vocab, "weight sharing needs one vocabulary for both sides"
        model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
        model.generator.proj.weight = model.tgt_embed[0].lut.weight
    for p in model.parameters():                                               # Xavier initialisation
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------------------------------------------
def subsequent_mask(size):
    """Lower triangle True (may look), upper triangle False (hidden)."""
    return torch.triu(torch.ones(1, size, size), diagonal=1) == 0


def make_masks(src, tgt_in, pad_id=PAD):
    """src: (B, n_src) token ids. tgt_in: (B, n_tgt) the decoder input, i.e. the target shifted right by one.
    Returns src_mask (B, 1, n_src) and tgt_mask (B, n_tgt, n_tgt): padding mask AND causal mask."""
    src_mask = (src != pad_id).unsqueeze(-2)
    tgt_mask = (tgt_in != pad_id).unsqueeze(-2) & subsequent_mask(tgt_in.size(-1)).to(tgt_in.device)
    return src_mask, tgt_mask


# ----------------------------------------------------------------------------------------------------------------
# Training: label-smoothed loss, learning-rate schedule, one step
# ----------------------------------------------------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """Cross-entropy on log-probabilities, with the target distribution (1 - eps) on the right token and eps spread
    evenly over the vocabulary; positions whose target is padding are ignored."""
    def __init__(self, smoothing=0.1, pad_id=PAD):
        super().__init__()
        self.smoothing, self.pad_id = smoothing, pad_id

    def forward(self, log_probs, target):                          # (B, n, V), (B, n)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        uniform = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * uniform
        keep = target != self.pad_id
        return (loss * keep).sum() / keep.sum()


def rate(step, model_size=512, factor=1.0, warmup=4000):
    step = max(step, 1)                                            # avoid 0 to a negative power
    return factor * model_size ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def train_step(model, optimizer, scheduler, criterion, src, tgt, pad_id=PAD):
    """tgt includes <Begin> ... <end>. The decoder input is tgt[:, :-1], the answer is tgt[:, 1:]."""
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    src_mask, tgt_mask = make_masks(src, tgt_in, pad_id)
    log_probs = model.generator(model(src, tgt_in, src_mask, tgt_mask))
    loss = criterion(log_probs, tgt_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


# ----------------------------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol=BOS):
    """Greedy decoding for a whole batch. The encoder runs once; the decoder re-reads everything generated so far at
    every step (the causal mask keeps each earlier position computed as in training). Returns (B, max_len),
    starting with <Begin>."""
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=src.dtype)
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).to(src.device))
        next_word = model.generator(out[:, -1]).argmax(dim=-1, keepdim=True)   # last position only
        ys = torch.cat([ys, next_word], dim=1)
    return ys


# ----------------------------------------------------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_self_tests():
    torch.manual_seed(0)
    print("self-tests")

    # 1. shapes
    m = make_model(50, 50, N=2, d_model=64, d_ff=128, h=4, dropout=0.0).eval()
    src = torch.randint(3, 50, (2, 7)); tgt_in = torch.randint(3, 50, (2, 5))
    src_mask, tgt_mask = make_masks(src, tgt_in)
    memory = m.encode(src, src_mask)
    out = m(src, tgt_in, src_mask, tgt_mask)
    assert memory.shape == (2, 7, 64) and out.shape == (2, 5, 64) and m.generator(out).shape == (2, 5, 50)
    assert src_mask.shape == (2, 1, 7) and tgt_mask.shape == (2, 5, 5)
    print("  ok  shapes: memory (B, n_src, d), decoder output (B, n_tgt, d), log-probs (B, n_tgt, V)")

    # 2. the causal mask is lower-triangular
    expected = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    assert (subsequent_mask(5)[0].long() == expected).all()
    print("  ok  subsequent_mask(5) is the lower triangle")

    # 3. no peeking: changing a later decoder input leaves every earlier output unchanged
    tgt_b = tgt_in.clone(); tgt_b[:, 3:] = torch.randint(3, 50, (2, 2))
    out_b = m(src, tgt_b, src_mask, make_masks(src, tgt_b)[1])
    assert torch.allclose(out[:, :3], out_b[:, :3], atol=1e-6)
    assert not torch.allclose(out[:, 3:], out_b[:, 3:], atol=1e-6)
    print("  ok  causal mask: changing positions 3-4 leaves outputs at positions 0-2 unchanged")

    # 4. the attention weights of masked positions are exactly zero
    w = m.decoder.layers[0].self_attn.attn                          # (B, h, n, n)
    assert (w.masked_select(~tgt_mask.unsqueeze(1).expand_as(w)) == 0).all()
    print("  ok  attention weights above the diagonal are exactly 0")

    # 5. padding does not change the result
    src_pad = torch.cat([src, torch.zeros(2, 3, dtype=src.dtype)], dim=1)
    mem_pad = m.encode(src_pad, make_masks(src_pad, tgt_in)[0])[:, :7]
    assert torch.allclose(memory, mem_pad, atol=1e-5)
    print("  ok  padding: appending <pad> to the source leaves the encoder output at real positions unchanged")

    # 6. -inf versus multiplying by 0 (the table in the blog post)
    row = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])                    # scores after "multiply by 0"
    masked = torch.tensor([2.0, 1.0, -1e9, -1e9, -1e9])              # scores after masked_fill
    p0, pm = row.softmax(0), masked.softmax(0)
    assert abs(p0[2:].sum().item() - 0.229) < 1e-3 and pm[2:].sum().item() == 0.0
    print("  ok  multiply by 0 leaves %.1f%% of the weight on hidden positions; -inf leaves 0%%" % (100 * p0[2:].sum()))

    # 7. parameter counts of the paper's base model with a 37000-token vocabulary
    big = make_model(37000, 37000, tie_weights=False)
    n_untied = count_params(big); del big
    big = make_model(37000, 37000, tie_weights=True)
    n_tied = count_params(big)
    enc = big.encoder.layers[0]
    ffn = count_params(enc.feed_forward); att = count_params(enc.self_attn)
    del big
    assert 100e6 < n_untied < 102e6 and 62e6 < n_tied < 64e6
    print("  ok  base model, vocabulary 37000: %.1f M parameters untied, %.1f M with the three matrices tied "
          "(paper: 65 M)" % (n_untied / 1e6, n_tied / 1e6))
    print("  ok  one encoder block: feed-forward %.2f M, attention %.2f M (%.0f%% feed-forward)"
          % (ffn / 1e6, att / 1e6, 100 * ffn / (ffn + att)))

    # 8. positional-encoding periods quoted in the blog post
    periods = {d: 2 * math.pi * 10000 ** (d / 512) for d in (0, 16, 64, 256, 510)}
    assert [round(periods[d]) for d in (0, 16, 64, 256)] == [6, 8, 20, 628] and 60000 < periods[510] < 61000
    print("  ok  position-encoding periods: dim 0 -> %.1f, 16 -> %.1f, 64 -> %.1f, 256 -> %.0f, 510 -> %.0f positions"
          % tuple(periods[d] for d in (0, 16, 64, 256, 510)))


# ----------------------------------------------------------------------------------------------------------------
# Toy task: copy a random sequence
# ----------------------------------------------------------------------------------------------------------------
def copy_batch(batch, vocab, min_len=3, max_len=8):
    """Source: random tokens (ids 3 .. vocab-1), padded. Target: <Begin> + the same tokens + <end>, padded."""
    lens = torch.randint(min_len, max_len + 1, (batch,))
    src = torch.zeros(batch, max_len, dtype=torch.long)
    tgt = torch.zeros(batch, max_len + 2, dtype=torch.long)
    for i, n in enumerate(lens.tolist()):
        seq = torch.randint(3, vocab, (n,))
        src[i, :n] = seq
        tgt[i, 0], tgt[i, 1:n + 1], tgt[i, n + 1] = BOS, seq, EOS
    return src, tgt


def evaluate(model, vocab, n=200):
    model.eval()
    src, tgt = copy_batch(n, vocab)
    src_mask = (src != PAD).unsqueeze(-2)
    ys = greedy_decode(model, src, src_mask, max_len=tgt.size(1))
    exact = 0
    for i in range(n):
        gold = tgt[i].tolist(); gold = gold[:gold.index(EOS) + 1]
        got = ys[i].tolist()[:len(gold)]
        exact += int(got == gold)
    model.train()
    return exact / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if not args.skip_tests:
        run_self_tests()

    torch.manual_seed(args.seed)
    vocab = 12
    model = make_model(vocab, vocab, N=2, d_model=64, d_ff=128, h=4, dropout=0.1, tie_weights=True)
    print("\ncopy task: vocabulary %d, model with %d parameters (2 layers, d_model 64, 4 heads, weights tied)"
          % (vocab, count_params(model)))
    print("before training: exact-match accuracy %.1f%%" % (100 * evaluate(model, vocab)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: rate(s, model_size=64, factor=1.0, warmup=100))
    criterion = LabelSmoothingLoss(smoothing=0.1)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        src, tgt = copy_batch(args.batch, vocab)
        loss = train_step(model, optimizer, scheduler, criterion, src, tgt)
        if step == 1 or step % 100 == 0 or step == args.steps:
            print("  step %4d  loss %.3f  (%.0f s)" % (step, loss, time.time() - t0))
    acc = evaluate(model, vocab)
    print("after training: exact-match accuracy %.1f%% on 200 new sequences" % (100 * acc))
    model.eval()
    src, tgt = copy_batch(3, vocab)
    ys = greedy_decode(model, src, (src != PAD).unsqueeze(-2), max_len=tgt.size(1))
    for i in range(3):
        n = int((src[i] != PAD).sum())
        print("  source %s -> output %s" % (src[i, :n].tolist(), ys[i, 1:n + 1].tolist()))
    assert acc > 0.9, "the model did not learn the copy task; try more --steps"
    print("ok")


if __name__ == "__main__":
    main()
