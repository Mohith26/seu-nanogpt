# seu-nanogpt: what single-bit upsets do to nanoGPT training, and a cheap guard

Radiation in orbit flips bits (single-event upsets, SEUs). Starcloud trained Karpathy's nanoGPT on an H100 in orbit in late 2025, and Google's Suncatcher design paper ([arXiv 2511.19468](https://arxiv.org/abs/2511.19468)) notes that the impact of single-event effects on training jobs, and how well system-level mitigations work, still needs study. This repo is a small, fully reproducible software study of that question on a nanoGPT-style character model:

1. **Which bit flips actually hurt training?** Flip one bit of one float32 value in the training state (model parameters, AdamW first moment `exp_avg`, AdamW second moment `exp_avg_sq`) and measure the effect 200 steps later.
2. **Does a cheap detect-and-roll-back guard keep training on track** when upsets keep arriving?

Everything runs on a laptop CPU in about 25 minutes. No GPU, no paid services.

## Findings

From [`results/summary.md`](results/summary.md) (all numbers below are copied from it):

**One flip (384 trials: 3 state buffers x 32 bit positions x 4 random elements).** 372 of 384 flips were fully masked: validation loss 200 steps later was within 0.005 of the clean run. Every one of the other 12 wrecked the run, and they came from exactly three (buffer, bit) combinations, 4 of 4 trials each:

| buffer | bit | what happens |
|---|---|---|
| parameters | 30 (exponent MSB) | a weight becomes ~1e35 to 1e37; the loss goes NaN, or stays finite but the model collapses to a uniform guess (train loss 4.17 = ln 65) |
| AdamW `exp_avg` | 30 | the momentum entry becomes ~1e32 to 1e34; NaN in 3 of 4 trials, val loss +0.65 in the 4th |
| AdamW `exp_avg_sq` | 31 (sign) | a tiny positive value becomes a tiny negative one; `sqrt` of a negative number gives NaN on the next step |

Two things surprised me. The sign bit is harmless in parameters and in `exp_avg`, but it is the *only* deadly bit in `exp_avg_sq`, even though it changes the value by about 1e-7. And the exponent MSB is harmless in `exp_avg_sq`: a huge second moment just freezes that one weight's update. So 3 of 96 (buffer, bit) combinations, about 3.1% of uniformly random upsets in this state, kill the run, and the rest disappear.

**Training under a steady upset rate (6 seeds, 2000 steps, about 80 upsets per run).**

| seed | clean val | unguarded val | guarded val | guard rollbacks | replayed steps |
|---|---|---|---|---|---|
| 1 | 1.9010 | 2.4858 | 1.8992 | 1 | 3 |
| 2 | 1.9275 | 1.9269 | 1.9269 | 0 | 0 |
| 3 | 1.9137 | NaN (step 254) | 1.9326 | 2 | 42 |
| 4 | 1.9296 | NaN (step 485) | 1.9291 | 3 | 55 |
| 5 | 1.8977 | NaN (step 132) | 1.9034 | 2 | 34 |
| 6 | 1.9163 | NaN (step 1353) | 1.9192 | 5 | 118 |

- Without protection, 4 of 6 runs ended in NaN and one more finished silently damaged (+0.585 val loss with a finite loss curve).
- With the guard, all 6 runs finished within +0.019 of the clean run's val loss, at a cost of 0 to 118 replayed steps (at most 5.9% extra steps).
- The guard raised 0 false alarms on 6 fault-free runs, and those runs reproduced the clean val loss exactly.
- Median step time on fault-free runs went from 9.53 ms to 9.94 ms (+4.3%) with the guard on (Apple Silicon CPU, 4 threads; noisy at this model size).

## The guard

`seu.core.Guard`, about 60 lines. After every step it checks that the loss and gradient norm are finite and that the loss is not above 1.5x its running average (after step 100). Every 10 steps it checks that all parameters and both AdamW moments are finite and that `exp_avg_sq` is non-negative, which it must be by construction. It keeps the last two in-RAM snapshots (every 50 steps); on a trip it restores the newest one and replays, falling back to the older snapshot if the newest one trips again.

## Setup

- Model: nanoGPT's architecture (pre-LN blocks, causal attention, GELU MLP, tied embeddings) at nanoGPT's suggested CPU size: 4 layers, 4 heads, 128-dim embeddings, block size 64, batch 12, 0.80M parameters, AdamW (lr 1e-3, beta2 0.99, weight decay 0.1, 50 warmup steps), gradient clipping at 1.0.
- Data: tiny-shakespeare, character level, 90/10 split as in nanoGPT's `prepare.py`. Validation loss is the mean over 16 fixed batches.
- Fault model: flip one uniformly random bit (0 to 31) of one uniformly random float32 element. In experiment 2 upsets arrive as a Poisson process (mean gap 25 steps) and land anywhere in parameters + both moments, weighted by size. An upset is a one-time event: if the guard rolls back past it, it is not re-applied.
- Determinism: the same seed gives the same init, data order and upset schedule with and without the guard, and repeated clean continuations from a snapshot match bit for bit (checked in experiment 1 and in the tests).

## What this does and does not show

- The upset rate in experiment 2 is accelerated by many orders of magnitude so that a 20-second run sees dozens of upsets. It is a stress test of the mitigation, not a prediction of orbital error rates.
- Flips are injected in software into fp32 tensors in CPU RAM. Real GPUs protect HBM and large SRAMs with ECC, so upsets that reach training state in practice come from less protected places (registers, logic, multi-bit events). The finding about *which* bits matter holds for any corruption that lands in these buffers; how often that happens on a given GPU is a hardware question this repo does not answer.
- No crashes, hangs or device resets (SEFIs) are modeled; those need checkpoint/restart at the job level, not this guard.
- One small model, fp32, one optimizer. bf16 or fp8 training, larger models and other optimizers may behave differently (for example, bf16 has 8 exponent bits too, so the exponent-MSB result should carry over, but that was not run).

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install torch numpy
.venv/bin/python -m unittest discover -s tests
.venv/bin/python experiments/single_fault.py      # ~15 min, resumable, writes results/single_fault.jsonl
.venv/bin/python experiments/guard.py             # ~9 min, resumable, writes results/guard.jsonl
python3 experiments/summarize.py                  # writes results/summary.md
```

Both experiment scripts take an optional time budget in seconds and resume where they stopped. Tested with Python 3.12, PyTorch 2.15.0 (CPU) on Apple Silicon.

`data/input.txt` is tiny-shakespeare from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
