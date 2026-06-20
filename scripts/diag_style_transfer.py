#!/usr/bin/env python3
"""A generalizing BEHAVIOR, not a lookup: a constrained output style
"always answer with the constant token c". It applies to inputs never shown, so
recall on HELD-OUT keys tests generalization, not memorization. We write the
style engram for c_X, inject it (recurrent only, neutral primer) into a recipient
holding a different style c_Y, and ask whether held-out queries now emit c_X.
Controls: no-transfer (recipient's own style) and a norm-matched random engram."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
REPS = int(os.environ.get("REPS", 3))
NTRAIN = int(os.environ.get("NTRAIN", 8))
NTEST = int(os.environ.get("NTEST", 8))
SEEDS = int(os.environ.get("SEEDS", 40))
AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VALS = list("0123456789")
PRIMER = "Q=Q;Z=Z;W=W;"

tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id


def ids(t, bos=False):
    x = tok(t, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + x if bos else x], device="cuda")


def getrec(c): return [c[i]["recurrent_state"].detach().clone() for i in range(len(c))]
def setrec(c, st):
    for i in range(len(c)): c[i]["recurrent_state"] = st[i]
def fwd(x, cache=None):
    with torch.no_grad(): return model(input_ids=x, past_key_values=cache, use_cache=True)
def add(a, b): return [x + y for x, y in zip(a, b)]


def demos(keys, c):
    return "".join("%s=%s;" % (k, c) for k in keys) * REPS


def engram(keys, c):
    return getrec(fwd(ids(demos(keys, c), bos=True)).past_key_values)


def random_like(e, r=16):
    out = []
    for s in e:
        k = min(r, s.shape[-1]); g = torch.Generator(device=s.device)
        g.manual_seed(int(s.float().abs().sum().item() * 1e3) % (2 ** 31))
        A = torch.randn(*s.shape[:-1], k, device=s.device, dtype=torch.float32, generator=g)
        B = torch.randn(*s.shape[:-2], k, s.shape[-1], device=s.device, dtype=torch.float32, generator=g)
        M = A @ B
        out.append((M * (s.float().norm() / (M.norm() + 1e-9))).to(s.dtype))
    return out


def emits(state, keys, c):
    ok = 0
    for k in keys:
        carrier = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(carrier, state)
        o = fwd(ids("%s=" % k), cache=carrier)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == c
    return ok / len(keys)


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


clean, notrans, rnd, full = [], [], [], []
for s in range(SEEDS):
    rng = random.Random(3000 + s)
    cX, cY = rng.sample(VALS, 2)
    L = rng.sample(list(AL), NTRAIN + NTEST)
    trainX, test = L[:NTRAIN], L[NTRAIN:]
    trainY = rng.sample(list(AL), NTRAIN)
    eX = engram(trainX, cX)
    eY = engram(trainY, cY)
    clean.append(emits(eX, test, cX))                 # held-out: does style cX generalize
    notrans.append(emits(eY, test, cX))               # recipient has cY -> should NOT emit cX
    rnd.append(emits(add(eY, random_like(eX)), test, cX))
    full.append(emits(add(eY, eX), test, cX))         # TRANSFER: held-out now emit cX?

print("model:", mid, " style=constant-output  reps=%d ntrain=%d ntest=%d seeds=%d (chance~0.1)"
      % (REPS, NTRAIN, NTEST, SEEDS))
for name, d in [("clean held-out (ceiling)", clean), ("no-transfer held-out", notrans),
                ("random held-out", rnd), ("FULL TRANSFER held-out", full)]:
    m, lo, hi = boot(d)
    print("  %-30s %.3f  [%.3f, %.3f]" % (name, m, lo, hi))
for nm, a, b in [("full - notransfer", full, notrans), ("full - random", full, rnd)]:
    df = np.array(a) - np.array(b); rng = np.random.default_rng(0)
    bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
    print("  %-22s d=%.3f [%.3f, %.3f] p=%.4f" %
          (nm, df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
