#!/usr/bin/env python3
"""
Distributed Engrammics -- scientific experiment harness.

This is a pre-registered, falsifiable test of the core claim: a low-rank
fast-weight state delta ("engram"), extracted by reading a skill, can be added
WITHOUT GRADIENTS to another agent's fast-weight state to transfer the skill
specifically and compositionally; the same algebra supports targeted forgetting
and subspace governance.

Two backends share ONE statistical engine:
  - toy : NumPy associative-memory model (runs anywhere; used to validate the
          harness itself and to prove the algebraic claims).
  - lm  : a DeltaNet checkpoint via flash-linear-attention; the fast-weight
          state F is the per-layer recurrent state. This is the real test.

------------------------------------------------------------------------------
PRE-REGISTERED HYPOTHESES AND DECISION RULES (paired bootstrap over seeds,
95% CIs, Holm-Bonferroni across primary one-sided tests, alpha = 0.05):

  H1  Specific transfer (PRIMARY).
      transfer_full beats BOTH controls:
        (a) no-transfer control  (B holds only Y, queried on X)
        (b) random-engram control (B + random rank-matched engram)
      Rule: CI lower bound of each paired difference > 0, AND transfer_full
      mean > chance + 0.15. If clean-write ceiling itself <= chance + 0.15 the
      task is unlearnable by this model -> INCONCLUSIVE (not a refutation).

  H2  Non-interference. Adding X to B does not destroy B's pre-existing Y.
      Rule (equivalence): CI lower bound of (recall_Y_after - recall_Y_before)
      > -0.10.

  H3  Targeted forgetting. Subtracting X's engram from a state that read X then
      Y drops recall_X while preserving recall_Y.
      Rule: CI lower bound of (recall_X_joint - recall_X_forget) > 0 AND
      CI lower bound of (recall_Y_forget - recall_Y_joint) > -0.10.

  H4  Governance. Integrating X through the ADMITTED subspace recovers X;
      through the DENIED subspace it does not.
      Rule: CI lower bound of (recall_admit - recall_deny) > 0.

  H5  Rank dose-response. recall_X increases with the transferred rank r.
      Rule: CI lower bound of (recall at r_max - recall at r_min) > 0 AND the
      per-rank means are non-decreasing.

Exit code: 0 iff the PRIMARY hypothesis H1 is SUPPORTED; 3 if INCONCLUSIVE
(model cannot do the task); 1 otherwise. Secondary hypotheses are reported but
do not change the exit code.
------------------------------------------------------------------------------
"""
import argparse
import os
import sys
from collections import namedtuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Skill = namedtuple("Skill", "data")


# ============================================================================
# Statistics
# ============================================================================
def boot_mean_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_boot(a, b, n_boot=10000, seed=0):
    """Paired bootstrap of mean(a - b). Returns mean, ci_lo, ci_hi, and the
    bootstrap distribution of the difference."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    bd = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)), bd


def p_greater(bd):                 # H0: diff <= 0
    return float((bd <= 0).mean())


def p_equiv(bd, margin):           # H0: diff <= -margin  (i.e. a real drop)
    return float((bd <= -margin).mean())


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals); keep = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            keep[i] = True
        else:
            break
    return keep


def verdict_gt0(lo, hi):
    return "SUPPORTED" if lo > 0 else ("REFUTED" if hi < 0 else "INCONCLUSIVE")


def verdict_equiv(lo, margin):
    return "SUPPORTED" if lo > -margin else "NOT SUPPORTED"


# ============================================================================
# Backend interface
# ============================================================================
class Backend:
    rank_grid = (1, 2, 4, 8)
    r_ref = 8
    def chance(self): raise NotImplementedError
    def skill(self, seed): raise NotImplementedError
    def engram(self, skill): raise NotImplementedError           # write from zero
    def write_into(self, state, skill): raise NotImplementedError  # continue writing
    def recall(self, state, skill): raise NotImplementedError
    def add(self, a, b): raise NotImplementedError
    def sub(self, a, b): raise NotImplementedError
    def low_rank(self, e, r): raise NotImplementedError
    def random_like(self, e, r): raise NotImplementedError
    def govern(self, e, mode, r): raise NotImplementedError       # mode in {admit, deny}


# ---------------------------------------------------------------------------
# Toy backend (NumPy). Validates the harness and proves the algebra.
# ---------------------------------------------------------------------------
class ToyBackend(Backend):
    rank_grid = (1, 2, 4, 8)
    r_ref = 8

    def __init__(self):
        from engrammics_proto import (init_conatus, train_conatus, sample_skill,
                                       build_engram, encode_keys, read,
                                       recall_metrics, l2norm, delta_write,
                                       N_PAIRS, D_K, D_V)
        self._p = dict(sample_skill=sample_skill, encode_keys=encode_keys,
                       read=read, recall_metrics=recall_metrics, l2norm=l2norm,
                       delta_write=delta_write, build_engram=build_engram)
        self.N = N_PAIRS; self.D_K = D_K; self.D_V = D_V
        self.Wk = train_conatus(init_conatus())

    def chance(self): return 1.0 / self.N

    def skill(self, seed):
        cues, resp = self._p["sample_skill"](np.random.default_rng(seed))
        return Skill((cues, resp))

    def _K(self, skill): return self._p["encode_keys"](self.Wk, skill.data[0])

    def engram(self, skill):
        return self._p["build_engram"](self.Wk, skill.data[0], skill.data[1])[0]

    def write_into(self, state, skill):
        return self._p["delta_write"](state, self._K(skill), skill.data[1])

    def recall(self, state, skill):
        o = self._p["read"](state, self._K(skill))
        return self._p["recall_metrics"](o, skill.data[1])[0]

    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b

    def low_rank(self, e, r):
        U, s, Vt = np.linalg.svd(e, full_matrices=False)
        r = min(r, len(s))
        return (U[:, :r] * s[:r]) @ Vt[:r]

    def random_like(self, e, r):
        rng = np.random.default_rng(int(abs(e).sum() * 1e3) % (2**31))
        A = rng.standard_normal((e.shape[0], r)); B = rng.standard_normal((r, e.shape[1]))
        M = A @ B
        return M * (np.linalg.norm(e) / (np.linalg.norm(M) + 1e-12))

    def govern(self, e, mode, r):
        U = np.linalg.svd(e, full_matrices=False)[0]
        r = min(r, U.shape[1]); Ur = U[:, :r]; P = Ur @ Ur.T
        if mode == "deny":
            P = np.eye(P.shape[0]) - P
        return P @ e


# ---------------------------------------------------------------------------
# LM backend (DeltaNet via flash-linear-attention). The real test.
# ---------------------------------------------------------------------------
class LMBackend(Backend):
    rank_grid = (1, 2, 4, 8, 16)
    r_ref = 10_000          # capped per head -> effectively full rank

    def __init__(self, model_id, device="auto", n_pairs=5):
        import random as _random
        import torch
        import fla  # noqa: F401  (registers architectures)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self._random = _random; self.n_pairs = n_pairs
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype).to(device).eval()

    # ---- cache / state plumbing (ADAPT _iter if your fla version differs) ----
    def _iter(self, cache):
        st = getattr(cache, "states", None)
        if isinstance(st, (list, tuple)):
            return list(st)
        return [cache[i] for i in range(len(cache))]

    def _get(self, cache):
        out = []
        for ls in self._iter(cache):
            rs = ls.get("recurrent_state") if isinstance(ls, dict) \
                else getattr(ls, "recurrent_state", None)
            if rs is None:
                raise RuntimeError("recurrent_state not found; adapt LMBackend._iter")
            out.append(rs.detach().clone())
        return out

    def _set(self, cache, states):
        for ls, rs in zip(self._iter(cache), states):
            if isinstance(ls, dict):
                ls["recurrent_state"] = rs
            else:
                setattr(ls, "recurrent_state", rs)
        return cache

    def _next(self, text, cache=None):
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        pred = int(out.logits[:, -1, :].argmax(-1))
        return self.tok.decode([pred]).strip(), out.past_key_values

    def _cache_of(self, text):
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        return self.model(input_ids=ids, use_cache=True).past_key_values

    # ---- skill task: random injective letter -> digit map ----
    def chance(self):
        return 1.0 / 10.0

    def skill(self, seed):
        rng = self._random.Random(seed)
        keys = rng.sample(list("ABCDEFGH"), self.n_pairs)
        vals = rng.sample([str(d) for d in range(10)], self.n_pairs)
        return Skill(list(zip(keys, vals)))

    def _demos(self, skill):
        return "".join(f"{k}={v};" for k, v in skill.data)

    def engram(self, skill):
        with self.torch.no_grad():
            return self._get(self._cache_of(self._demos(skill)))

    def write_into(self, state, skill):
        with self.torch.no_grad():
            _, carrier = self._next("\n")
            self._set(carrier, state)
            ids = self.tok(self._demos(skill), return_tensors="pt").input_ids.to(self.device)
            c2 = self.model(input_ids=ids, past_key_values=carrier, use_cache=True).past_key_values
            return self._get(c2)

    def recall(self, state, skill):
        with self.torch.no_grad():
            ok = 0
            for k, v in skill.data:
                _, carrier = self._next("\n")
                self._set(carrier, state)
                pred, _ = self._next(f"{k}=", cache=carrier)
                ok += int(pred == v)
            return ok / len(skill.data)

    def add(self, a, b): return [x + y for x, y in zip(a, b)]
    def sub(self, a, b): return [x - y for x, y in zip(a, b)]

    def low_rank(self, e, r):
        out = []
        for s in e:
            U, sv, Vh = self.torch.linalg.svd(s.float(), full_matrices=False)
            k = min(r, sv.shape[-1])
            out.append(((U[..., :k] * sv[..., :k].unsqueeze(-2)) @ Vh[..., :k, :]).to(s.dtype))
        return out

    def random_like(self, e, r):
        out = []
        for s in e:
            k = min(r, s.shape[-1])
            A = self.torch.randn(*s.shape[:-1], k, device=s.device, dtype=self.torch.float32)
            B = self.torch.randn(*s.shape[:-2], k, s.shape[-1], device=s.device, dtype=self.torch.float32)
            M = A @ B
            M = M * (s.float().norm() / (M.norm() + 1e-9))
            out.append(M.to(s.dtype))
        return out

    def govern(self, e, mode, r):
        out = []
        for s in e:
            sf = s.float()
            U = self.torch.linalg.svd(sf, full_matrices=False)[0]
            k = min(r, U.shape[-1]); Ur = U[..., :k]
            P = Ur @ Ur.transpose(-1, -2)
            if mode == "deny":
                eye = self.torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
                P = eye - P
            out.append((P @ sf).to(s.dtype))
        return out


# ============================================================================
# Experiment
# ============================================================================
def run_experiment(be, n_seeds, base_seed=0, verbose=True):
    grid = be.rank_grid; rref = be.r_ref
    cols = ["clean", "notransfer", "random", "full", "Y_before", "Y_after",
            "joint_X", "joint_Y", "forget_X", "forget_Y", "admit", "deny"]
    cols += [f"r{r}" for r in grid]
    acc = {c: [] for c in cols}

    for i in range(n_seeds):
        sx = be.skill(base_seed + 10_000 + i)
        sy = be.skill(base_seed + 20_000 + i)
        eX = be.engram(sx); eY = be.engram(sy); baseY = eY

        acc["clean"].append(be.recall(eX, sx))                      # ceiling
        acc["notransfer"].append(be.recall(baseY, sx))              # control a
        acc["random"].append(be.recall(be.add(baseY, be.random_like(eX, rref)), sx))  # control b
        acc["full"].append(be.recall(be.add(baseY, eX), sx))        # transfer (full)
        for r in grid:
            acc[f"r{r}"].append(be.recall(be.add(baseY, be.low_rank(eX, r)), sx))

        acc["Y_before"].append(be.recall(baseY, sy))                # H2
        acc["Y_after"].append(be.recall(be.add(baseY, eX), sy))

        joint = be.write_into(eX, sy)                               # read X then Y (real)
        forget = be.sub(joint, eX)                                  # remove X's engram
        acc["joint_X"].append(be.recall(joint, sx))
        acc["joint_Y"].append(be.recall(joint, sy))
        acc["forget_X"].append(be.recall(forget, sx))
        acc["forget_Y"].append(be.recall(forget, sy))

        acc["admit"].append(be.recall(be.add(baseY, be.govern(eX, "admit", rref)), sx))  # H4
        acc["deny"].append(be.recall(be.add(baseY, be.govern(eX, "deny", rref)), sx))

        if verbose:
            print(f"  seed {i+1}/{n_seeds} done", flush=True)
    return acc, grid


def evaluate(be, acc, grid):
    chance = be.chance()
    def m(c): return boot_mean_ci(acc[c])

    print("\n" + "=" * 72)
    print(f" CONDITION MEANS  (95% bootstrap CI)        chance = {chance*100:.1f}%")
    print("=" * 72)
    order = ["clean", "notransfer", "random"] + [f"r{r}" for r in grid] + \
            ["full", "Y_before", "Y_after", "joint_X", "joint_Y",
             "forget_X", "forget_Y", "admit", "deny"]
    for c in order:
        mean, lo, hi = m(c)
        print(f"  {c:<11} {mean*100:6.1f}%   [{lo*100:5.1f}, {hi*100:5.1f}]")

    # ---- hypothesis tests ----
    print("\n" + "=" * 72)
    print(" HYPOTHESIS TESTS  (paired bootstrap, 95% CI, Holm-corrected primary)")
    print("=" * 72)

    ceiling = np.mean(acc["clean"])
    inconclusive = ceiling <= chance + 0.15

    d1a = paired_boot(acc["full"], acc["notransfer"])
    d1b = paired_boot(acc["full"], acc["random"])
    d4 = paired_boot(acc["admit"], acc["deny"])
    rmax, rmin = f"r{grid[-1]}", f"r{grid[0]}"
    d5 = paired_boot(acc[rmax], acc[rmin])
    primary = {"H1a full>no-transfer": d1a, "H1b full>random": d1b,
               "H4 admit>deny": d4, "H5 r_max>r_min": d5}
    pvals = [p_greater(d[3]) for d in primary.values()]
    keep = holm(pvals)

    rows = []
    for (name, d), pv, k in zip(primary.items(), pvals, keep):
        v = verdict_gt0(d[1], d[2])
        v = v + ("" if v != "SUPPORTED" else (" (Holm ok)" if k else " (Holm dropped)"))
        rows.append((name, d[0], d[1], d[2], pv, v))

    # equivalence / composite
    d2 = paired_boot(acc["Y_after"], acc["Y_before"])
    rows.append(("H2 Y preserved (equiv)", d2[0], d2[1], d2[2],
                 p_equiv(d2[3], 0.10), verdict_equiv(d2[1], 0.10)))
    d3drop = paired_boot(acc["joint_X"], acc["forget_X"])
    d3keep = paired_boot(acc["forget_Y"], acc["joint_Y"])
    h3 = "SUPPORTED" if (d3drop[1] > 0 and d3keep[1] > -0.10) else "NOT SUPPORTED"
    rows.append(("H3 forget X (drop)", d3drop[0], d3drop[1], d3drop[2],
                 p_greater(d3drop[3]), verdict_gt0(d3drop[1], d3drop[2])))
    rows.append(("H3 forget X (keep Y)", d3keep[0], d3keep[1], d3keep[2],
                 p_equiv(d3keep[3], 0.10), verdict_equiv(d3keep[1], 0.10)))

    print(f"  {'test':<26}{'Δ mean':>9}{'95% CI':>20}{'p':>8}  verdict")
    for name, mean, lo, hi, pv, v in rows:
        print(f"  {name:<26}{mean*100:8.1f}% [{lo*100:6.1f},{hi*100:6.1f}]{pv:8.3f}  {v}")

    mono = all(np.mean(acc[f"r{grid[j]}"]) <= np.mean(acc[f"r{grid[j+1]}"]) + 1e-9
               for j in range(len(grid) - 1))

    # ---- overall verdict on the PRIMARY claim (H1) ----
    print("\n" + "=" * 72)
    h1 = (d1a[1] > 0 and d1b[1] > 0 and keep[0] and keep[1]
          and np.mean(acc["full"]) > chance + 0.15)
    if inconclusive:
        print(" THEORY (H1): INCONCLUSIVE -- clean-write ceiling at/near chance.")
        print(" The model cannot perform the task; this is not a refutation.")
        code = 3
    elif h1:
        print(" THEORY (H1): SUPPORTED -- transfer beats both controls and chance.")
        code = 0
    else:
        print(" THEORY (H1): NOT SUPPORTED on this model/task.")
        code = 1
    print(f" Secondary: H2={rows[4][5]}  H3={h3}  H4={rows[2][5]}  "
          f"H5={'monotone' if mono else 'non-monotone'}/{rows[3][5]}")
    print("=" * 72)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["toy", "lm"], default="toy")
    ap.add_argument("--model", default=os.environ.get("MODEL_ID", ""))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print(f" DISTRIBUTED ENGRAMMICS -- scientific test   backend={args.backend}"
          f"   seeds={args.seeds}")
    print("=" * 72)

    if args.backend == "toy":
        be = ToyBackend()
    else:
        if not args.model:
            print("[FATAL] --backend lm requires --model or MODEL_ID")
            sys.exit(2)
        be = LMBackend(args.model, device=args.device, n_pairs=args.pairs)

    acc, grid = run_experiment(be, args.seeds, verbose=not args.quiet)
    code = evaluate(be, acc, grid)
    sys.exit(code)


if __name__ == "__main__":
    main()
