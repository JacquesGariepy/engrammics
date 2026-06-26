# Distributed Engrammics

**Gradient-free transfer of a persistent fast-weight associative state between agents.**

In linear-attention models (DeltaNet, RWKV-7) a skill learned in context lives in
the recurrent fast-weight state as a low-rank, additive object. This repo shows
you can *extract* that object from one agent, *add* it to another's state with no
gradient step, and have the second agent perform the skill, plus an algebra for
forgetting and subspace-gated governance. Full write-up: [`doc/2026-06-26-v1.0.0`](doc/2026-06-26-v1.0.0).

**Scope and honest limits.** This works only for architectures with a single
additive recurrent state (linear attention), **not** standard softmax
transformers, i.e. not most models deployed today. And we report what fails as
plainly as what works: on the real DeltaNet, gradient-free transfer, the rank
dose-response, and captured-key governance hold, but **two of the four operations
fail naively**: interference-free superposition (H2) damages the host skill, and
"surgical" forgetting (H3) erases the host along with the target. Both are
*recoverable*, but only as **lossy, tunable trade-offs** with the same key
projector (orthogonal injection preserves the host at a transfer-fidelity cost;
host-orthogonal forgetting preserves the host but removes the target only
partially). The full Proven / Not-proven ledger is the scorecard (Table 9) in the
paper.

---

## Reproduce the main result in one command

The controlled-regime result (paper Tables 1–2: gradient-free transfer separating
from both controls by ≈88 points, with a rank dose-response, non-interference,
targeted forgetting, and subspace governance) runs on **CPU in seconds** and needs
**only NumPy** (no GPU, no model download):

```bash
python src/engrammics_science.py --backend toy --seeds 60 --quiet
```

Expected output (abridged):

```
 CONDITION MEANS  (95% bootstrap CI)        chance = 12.5%
  clean        100.0%   notransfer 12.5%   random 12.1%
  r1 28.7%  r2 55.0%  r4 88.8%  r8 100.0%   <- rank dose-response
  full 100.0%   forget_X 0.2%  forget_Y 100.0%   admit 100.0%  deny 12.5%

 HYPOTHESIS TESTS  (paired bootstrap, 95% CI, Holm-corrected)
  H1a full>no-transfer   +87.5%  [84.8, 90.2]  p=0.000  SUPPORTED
  H1b full>random        +87.9%  [84.8, 90.8]  p=0.000  SUPPORTED
  H4  admit>deny         +87.5%  [84.8, 90.2]  p=0.000  SUPPORTED
  H5  r_max>r_min        +71.2%  [68.3, 74.0]  p=0.000  SUPPORTED
```

That is the whole claim, falsifiable and reproduced, in one line. The same harness
and the same verdict function are then pointed at a real language model.

## Reproduce the language-model headline

The headline LM result (paper Tables 5–6: a gradient-free engram added to a real
**DeltaNet-1.3B** recurrent state recovers the donor's skill at 68.7%, +61.3 over
no-transfer and +66.0 over a random engram, p<0.001; governance +62.0) needs a
GPU and the checkpoint:

```bash
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python src/engrammics_science.py --backend lm \
       --model ~/models/delta_net-1.3B-100B --seeds 30 \
       --dump results/perseed_lm_1.3B.csv
```

Expected output: the real verdicts, **including the two that fail** on the dense
model (this is what the scorecard reports, not just the wins):

```
 HYPOTHESIS TESTS  (DeltaNet-1.3B, 30 seeds, Holm-corrected)
  H1a full>no-transfer   +61.3%  SUPPORTED      <- gradient-free transfer
  H1b full>random        +66.0%  SUPPORTED
  H4  admit>deny         +62.0%  SUPPORTED      <- captured-key governance
  H5  r_max>r_min        +61.3%  SUPPORTED      <- rank dose-response
  H2  Y preserved        -26.0%  NOT SUPPORTED  <- superposition damages the host
  H3  forget X, keep Y   -71.3%  NOT SUPPORTED  <- forgetting is not surgical
```

H2 and H3 fail here because the two skills share key directions in a dense model;
they become host-safe trade-offs with the projector fixes (paper §6.6). One-time
setup (Python 3.12, venv, `pip install -r requirements.txt`, model
download) and a result-by-result command map for **every** table and figure are in
[`REPRODUCE.md`](REPRODUCE.md). The raw logs and per-seed CSVs behind every number
are in [`results/`](results/) (lab notebook: [`results/RESULTS.md`](results/RESULTS.md)).

## Reproduce the most novel result: transfer across distinct checkpoints

Moving an engram between two **genuinely distinct** RWKV-7-1.5B checkpoints
(`g1` → `world`), where naive injection is at chance and a closed-form, skill-
independent alignment (fit once per pair, both models frozen) recovers it:

```bash
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
A_ID=~/models/rwkv7-1.5B-g1 B_ID=~/models/rwkv7-1.5B-world \
  ANCHORS=24 SEEDS=25 python scripts/diag_xcheckpoint.py
```

Expected output:

```
  naive A->B (no alignment)   0.040  [0.008, 0.072]   <- chance
  aligned A->B (learned W)    0.968  [0.936, 0.992]   <- recipient's own ceiling
  aligned - naive   d=0.928   p=0.0000
```

Needs both RWKV-7-1.5B checkpoints (`fla-hub/rwkv7-1.5B-g1` and `-world`); see
[`REPRODUCE.md`](REPRODUCE.md) §3.

---

## Layout

| Path | What |
|---|---|
| `src/engrammics_science.py` | the harness: one statistical engine, two backends (`toy`, `lm`) judged by identical criteria |
| `src/engrammics_proto.py`   | NumPy associative-memory model (capacity, heterogeneity, sensitivity tables) |
| `scripts/`                  | per-experiment scripts; see `REPRODUCE.md` §5 for the paper-vs-exploratory split |
| `results/`                  | every raw log + per-seed CSVs |
| `doc/`                      | LaTeX source + compiled PDF |
| `REPRODUCE.md`              | environment, model checkpoints, and the full result→command→log map |

## Requirements

- One-command result above: Python 3.12 and NumPy (CPU only).
- Language-model results: an NVIDIA GPU (we used a single RTX 3090, 24 GB) and the
  pinned stack in `requirements.txt` (torch 2.12.1, flash-linear-attention 0.5.1,
  transformers 5.12.1). See `REPRODUCE.md`.
