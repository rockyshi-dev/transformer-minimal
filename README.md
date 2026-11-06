# transformer-minimal

The original encoder-decoder Transformer of [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
in one PyTorch file, with self-tests that check what a computer can check. It goes with the post
[The original Transformer, step by step](https://rockyshi.dev/posts/transformer-step-by-step/), which
explains every part and every tensor shape.

```bash
pip install torch                      # CPU is enough
python transformer_minimal.py          # self-tests, then a copy-task training run
python transformer_minimal.py --steps 600 --seed 1
```

## What the nine self-test lines check

1. shapes: memory, decoder output and log-probabilities;
2. `subsequent_mask(5)` is the lower triangle;
3. causal mask: changing decoder inputs at positions 3 and 4 leaves the outputs at positions 0 to 2 unchanged;
4. the attention weights above the diagonal are exactly 0;
5. padding: appending `<pad>` to the source leaves the encoder output at the real positions unchanged;
6. multiplying masked scores by 0 leaves 22.9% of the weight on hidden positions in the row [2, 1, 0, 0, 0]; setting them to -inf leaves 0%;
7. the paper's base model with a 37000-token vocabulary has 101.0 M parameters untied and 63.1 M with the source embedding, target embedding and output layer sharing one matrix (the paper reports 65 M);
8. one encoder block has 2.10 M parameters in the feed-forward network and 1.05 M in attention (67% feed-forward);
9. the position-encoding periods for dimensions 0, 16, 64, 256 and 510 (6.3, 8.4, 19.9, 628 and about 60,600 positions).

## The copy task

After the self-tests the script trains a small model (2 layers, d_model 64, 4 heads, weights tied, vocabulary of
12 tokens) to copy a random token sequence, with the Adam settings and the warm-up schedule of the paper and a
label-smoothed loss, and reports the exact-match accuracy of greedy decoding on 200 new sequences. It stops with
an error if the accuracy ends below 90%.

| Machine | PyTorch | Threads | 400 training steps | Whole script | Loss at step 400 | Exact match |
|---|---|---|---|---|---|---|
| x86 server CPU (Intel Xeon Gold 5320) | 2.13.0 | 4 | 13 s | 19 s and 27 s (two runs) | 0.617 | 99.5% |
| NanoPC-T4 (Rockchip RK3399, 6 ARM cores), busy with other jobs | 2.8.0 | 4 | 2,098 s | not timed | 0.635 | 98.0% |

PyTorch versions and CPUs draw different random numbers, so the exact figures differ between machines (seed 0); on one machine two runs gave identical losses and accuracy.

## What differs from The Annotated Transformer

The file is adapted from [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)
(Harvard NLP). The differences: the sub-layer wrapper is Pre-LN (the paper's Post-LN is included for comparison);
the source embedding, the target embedding and the output layer can share one matrix (`tie_weights=True`, which
needs one vocabulary for both sides); one function, `make_masks`, builds both masks from the decoder input; the
loss is a label-smoothed cross-entropy written out in a few lines; and the self-tests above.

## Licence

MIT, see `LICENSE`. The Annotated Transformer is also MIT-licensed; its notice is in `NOTICE`.
