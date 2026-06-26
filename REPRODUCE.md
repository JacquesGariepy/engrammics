# Reproducing Distributed Engrammics

This file is the single entry point to re-run every experiment in the paper. It
documents (a) the environment, (b) the model checkpoints (which live outside the
repo), and (c) a one-to-one map from every paper result to the script, command,
and output log that produced it.

The controlled/toy regime needs only NumPy on CPU. Everything labelled "LM"
needs an NVIDIA GPU; we used a single RTX 3090 (24 GB). The largest model
(RWKV-7-2.9B, bf16) needs ~6 GB of VRAM; the cross-checkpoint run loads two
1.5B models at once (~6 GB).

---

## 1. What is in the repo vs. what is not

In the repo:
- `src/`: the harness and the toy backend (3 files, all used; see below).
- `scripts/`: diagnostic scripts. Some produce paper numbers, some are
  development-only; the split is listed in §5.
- `results/`: every raw log and the per-seed CSVs that back the tables.
- `doc/`: the LaTeX source and compiled PDF.
- `requirements.txt`, this file.

NOT in the repo (recreate locally; see §2-§3):
- The Python virtual environment. We used `~/engr_venv` under WSL2.
- The model checkpoints. We stored them under `~/models/<repo-name>/`
  (~19 GB for the five needed checkpoints).

---

## 2. Environment

Tested on: WSL2 Ubuntu 24.04, Python 3.12.3, NVIDIA RTX 3090 (driver 591.86,
CUDA 13.1 runtime), torch 2.12.1+cu130, flash-linear-attention 0.5.1,
transformers 5.12.1, tokenizers 0.22.2, numpy 2.4.6.

```bash
python3 -m venv ~/engr_venv
source ~/engr_venv/bin/activate
pip install --upgrade pip
pip install torch==2.12.1            # pulls the matching CUDA wheels
pip install -r requirements.txt
```

For all LM runs we export, once per shell:

```bash
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

(The models are already on disk, so HF is not contacted. RWKV-7 needs
`trust_remote_code=True`, which the scripts pass automatically.)

---

## 3. Model checkpoints

All from the `fla-hub` organisation on the Hugging Face Hub. We stored each under
`~/models/<repo-name>/`. Download with the Hub CLI (or `git clone` + git-lfs):

```bash
pip install -U "huggingface_hub[cli]"
for R in delta_net-1.3B-100B delta_net-2.7B-100B \
         rwkv7-1.5B-g1 rwkv7-1.5B-world rwkv7-2.9B-g1 ; do
  huggingface-cli download "fla-hub/$R" --local-dir "$HOME/models/$R"
done
```

| Local path (`~/models/...`)    | HF repo                       | Used for |
|--------------------------------|-------------------------------|----------|
| `delta_net-1.3B-100B`          | `fla-hub/delta_net-1.3B-100B` | main LM tests, governance, H2/H3 fixes, multi-token, rule transfer (1.3B) |
| `delta_net-2.7B-100B`          | `fla-hub/delta_net-2.7B-100B` | replication across model size; rule transfer (2.7B) |
| `rwkv7-1.5B-g1`                | `fla-hub/rwkv7-1.5B-g1`       | second-architecture replication; cross-checkpoint **donor** |
| `rwkv7-1.5B-world`             | `fla-hub/rwkv7-1.5B-world`    | cross-checkpoint **recipient** |
| `rwkv7-2.9B-g1`                | `fla-hub/rwkv7-2.9B-g1`       | in-context rule induction + arbitrary-rule (Caesar) transfer |

Not needed for any paper result: `fla-hub/delta_net-1.3B-8K-100B` was tried as a
cross-checkpoint recipient but cannot perform the recall task at all (its own
engram recalls at 0.0), so the pair was discarded. Do not download it.

The toy/controlled regime needs no checkpoint.

---

## 4. Result-by-result reproduction map

Run from the repo root with the venv active. `M13`, `M27`, `RW15G`, `RW15W`,
`RW29` are shorthand for the model paths:

```bash
M13=~/models/delta_net-1.3B-100B ; M27=~/models/delta_net-2.7B-100B
RW15G=~/models/rwkv7-1.5B-g1 ; RW15W=~/models/rwkv7-1.5B-world
RW29=~/models/rwkv7-2.9B-g1
```

### Controlled regime (CPU, no GPU)
| Paper element | Command | Output log |
|---|---|---|
| Tables `tab:means`, `tab:tests` (toy, 60 seeds) | `python src/engrammics_science.py --backend toy --seeds 60 --quiet` | `results/stage_a_science_toy.log` |
| Tables `tab:capacity`, `tab:hetero`, sensitivity (20 seeds); gradient check | `python src/engrammics_proto.py` | `results/proto_demonstrations.log` |

### Language model: main protocol (DeltaNet)
| Paper element | Command (prefix with the two exports + the `M*` defs) | Output |
|---|---|---|
| Tables `tab:lm-means`, `tab:lm-tests` (1.3B, 30 seeds) + per-seed CSV | `python src/engrammics_science.py --backend lm --model $M13 --seeds 30 --dump results/perseed_lm_1.3B.csv` | `results/stage_b_science_lm.log`, `results/perseed_lm_1.3B.csv` |
| Robustness across model size (2.7B, 20 seeds) + per-seed CSV | `python src/engrammics_science.py --backend lm --model $M27 --seeds 20 --dump results/perseed_lm_2.7B.csv` | `results/stage_b_science_lm_2.7B.log`, `results/perseed_lm_2.7B.csv` |
| Reps ablation (reps = 1, 2, 3; 15 seeds each) | `for R in 1 2 3; do python src/engrammics_science.py --backend lm --model $M13 --seeds 15 --reps $R --quiet; done` | `results/stage_b_reps_ablation.log` |

### Degradations and the controllable fixes (DeltaNet-1.3B)
| Paper element | Command | Output |
|---|---|---|
| Table `tab:disjoint` (3-level overlap) | `SEEDS=16 python scripts/diag_disjoint2.py $M13` | `results/stage_b_disjoint2.log` |
| Structured controls (shuffled-values, wrong-skill) | `SEEDS=30 python scripts/diag_controls.py $M13` | `results/stage_b_controls.log` |
| H2 fix: orthogonal injection | `SEEDS=30 python scripts/diag_ortho_inject.py $M13` | `results/stage_b_ortho_inject.log` |
| H3 fix: Y-safe forgetting | `SEEDS=24 python scripts/diag_h3_fix.py $M13` | `results/stage_b_h3_fix.log` |
| Multi-token associations + shuffled-key control | `SEEDS=40 python scripts/diag_multitoken.py $M13` | `results/stage_b_multitoken.log` |

### Memory or skill? (rule transfer)
| Paper element | Command | Output |
|---|---|---|
| Caesar induction probe, DeltaNet 1.3B + 2.7B | `python scripts/diag_skill.py $M13` ; `python scripts/diag_skill.py $M27` | `results/stage_b_rule_probe.log` |
| Concept (vowel/consonant, half) induction probe | `python scripts/diag_concept.py $M13` ; `python scripts/diag_concept.py $M27` | `results/stage_b_concept_probe.log` |
| Table `tab:rule`, 1.3B column (35 valid of 40 seeds) | `SEEDS=40 python scripts/diag_rule_transfer.py $M13` | `results/stage_b_rule_transfer.log` |
| Table `tab:rule`, 2.7B column (52 valid of 60 seeds) | `SEEDS=60 python scripts/diag_rule_transfer.py $M27` | `results/stage_b_rule_transfer_2.7B.log` |
| Constant output style transfer (40 seeds) | `SEEDS=40 python scripts/diag_style_transfer.py $M13` | `results/stage_b_style_transfer.log` |
| Second-architecture replication (RWKV-7-1.5B, 52/60 seeds) | `SEEDS=60 python scripts/diag_rule_transfer.py $RW15G` | `results/stage_b_rule_transfer_rwkv1.5B.log` |
| RWKV-7 induction probes (1.5B, 2.9B) | `python scripts/diag_skill.py $RW15G` ; `python scripts/diag_skill.py $RW29` ; same with `diag_concept.py` | `results/stage_b_rule_probe_rwkv1.5B.log`, `..._rwkv2.9B.log` |
| Arbitrary induced rule (Caesar k=2) transfer, RWKV-7-2.9B (50 seeds) | `K=2 SEEDS=50 python scripts/diag_caesar_transfer.py $RW29` | `results/stage_b_caesar_transfer_rwkv2.9B.log` |
| Cross-checkpoint transfer g1 -> world (24 anchors, 25 seeds) | `A_ID=$RW15G B_ID=$RW15W ANCHORS=24 SEEDS=25 python scripts/diag_xcheckpoint.py` | `results/stage_b_xcheckpoint_rwkv.log` |

Notes:
- The rule-transfer scripts skip seeds whose train set lacks both classes, so the
  number of *valid* seeds is slightly below `SEEDS` (e.g. 60 -> 52).
- `random` controls are seeded deterministically from the engram contents, so
  they reproduce exactly; the rest is reproducible up to CUDA kernel
  nondeterminism. `base_seed = 0` throughout the main harness.

---

## 5. Script inventory: paper vs. exploratory

**Used to produce paper numbers** (see §4): `src/engrammics_science.py`,
`src/engrammics_proto.py`, and the scripts
`diag_disjoint2.py`, `diag_controls.py`, `diag_ortho_inject.py`,
`diag_h3_fix.py`, `diag_multitoken.py`, `diag_skill.py`, `diag_concept.py`,
`diag_rule_transfer.py`, `diag_style_transfer.py`, `diag_caesar_transfer.py`,
`diag_xcheckpoint.py`.

**Setup / inspection aids** (not run for numbers, but document the cache layout
the harness depends on): `scripts/inspect_cache.py` (DeltaNet cache),
`scripts/inspect_rwkv.py` (RWKV-7 cache), `src/pick_model.py` (discovers a pure
DeltaNet checkpoint on the Hub), `scripts/run_engrammics.sh` (one-click launcher
for the toy gate + DeltaNet auto-discovery; predates the RWKV and diagnostic
work, so it does not cover §4 in full).

**Exploratory / development-only** (kept for provenance; no paper result depends
on them, safe to ignore when reproducing): `diag_fix.py` (tested and *refuted*
a double-baseline hypothesis), `diag_formats.py`, `diag_inject.py`,
`diag_primer.py`, `diag_recall.py`, `diag_keys.py`, `diag_skill_transfer.py`,
`diag_disjoint.py` (the 2-level precursor superseded by `diag_disjoint2.py`).
The two committed `results/.diag_*.out` dot-files are likewise stale scratch
output and are not cited by the paper.

For provenance of the main numbers and the exact environment, see also the
"Provenance (pour audit)" block at the top of `results/RESULTS.md`.
