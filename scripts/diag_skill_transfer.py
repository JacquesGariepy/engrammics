#!/usr/bin/env python3
"""Skill vs dictionary: does a transferred engram carry a GENERALIZING RULE?
A skill is a Caesar shift f(x)=x+k. We write its engram from demos on a TRAIN
set of letters, inject it (recurrent state only, neutral primer) into a recipient
that holds a different skill Y, and test on HELD-OUT letters never shown. If full
transfer beats the no-transfer and random controls ON HELD-OUT items, the engram
transferred a rule, not a lookup table. Same inject mechanism as the main harness.

Env: K (shift, default 0=random per seed), REPS (default 3), SEP (default '>'),
NTRAIN (default 8), NTEST (default 8), SEEDS (default 40)."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
REPS = int(os.environ.get("REPS", 3))
SEP = os.environ.get("SEP", ">")
NTRAIN = int(os.environ.get("NTRAIN", 8))
NTEST = int(os.environ.get("NTEST", 8))
SEEDS = int(os.environ.get("SEEDS", 40))
KFIX = int(os.environ.get("K", 0))
AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRIMER = "Q>Q;Z>Z;W>W;"            # neutral, same '>' format, keys outside typical draw

tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id


def ids(text, bos=False):
    t = tok(text, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + t if bos else t], device="cuda")


def getrec(c):
    return [c[i]["recurrent_state"].detach().clone() for i in range(len(c))]


def setrec(c, st):
    for i in range(len(c)):
        c[i]["recurrent_state"] = st[i]


def fwd(x, cache=None):
    with torch.no_grad():
        return model(input_ids=x, past_key_values=cache, use_cache=True)


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def caesar(x, k):
    return AL[(AL.index(x) + k) % 26]


def demos(pairs):
    return "".join("%s%s%s;" % (a, SEP, b) for a, b in pairs) * REPS


def engram(pairs):
    return getrec(fwd(ids(demos(pairs), bos=True)).past_key_values)


def random_like(e, r=16):
    out = []
    for s in e:
        k = min(r, s.shape[-1])
        g = torch.Generator(device=s.device)
        g.manual_seed(int(s.float().abs().sum().item() * 1e3) % (2 ** 31))
        A = torch.randn(*s.shape[:-1], k, device=s.device, dtype=torch.float32, generator=g)
        B = torch.randn(*s.shape[:-2], k, s.shape[-1], device=s.device, dtype=torch.float32, generator=g)
        M = A @ B
        out.append((M * (s.float().norm() / (M.norm() + 1e-9))).to(s.dtype))
    return out


def recall(state, test_letters, k):
    ok = 0
    for x in test_letters:
        carrier = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(carrier, state)
        o = fwd(ids("%s%s" % (x, SEP)), cache=carrier)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == caesar(x, k)
    return ok / len(test_letters)


def make(seed):
    rng = random.Random(seed)
    k = KFIX if KFIX else rng.randint(1, 9)
    ky = (k + rng.randint(1, 9))
    ky = ky if (ky % 26) else k + 1
    L = rng.sample(list(AL), NTRAIN + NTEST)
    trainX, test = L[:NTRAIN], L[NTRAIN:]
    Ly = rng.sample(list(AL), NTRAIN)
    trainY = Ly
    X = [(a, caesar(a, k)) for a in trainX]
    Y = [(a, caesar(a, ky)) for a in trainY]
    return X, Y, test, k


def boot(d, n=10000):
    import numpy as np
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


clean, notrans, rnd, full, train_full = [], [], [], [], []
for s in range(SEEDS):
    X, Y, test, k = make(2000 + s)
    eX, eY = engram(X), engram(Y)
    clean.append(recall(eX, test, k))                       # ceiling on held-out
    notrans.append(recall(eY, test, k))                     # B holds Y, read X held-out
    rnd.append(recall(add(eY, random_like(eX)), test, k))   # random control
    full.append(recall(add(eY, eX), test, k))               # TRANSFER on held-out
    train_full.append(recall(add(eY, eX), [a for a, _ in X], k))  # transfer on TRAIN (memorized)

import numpy as np
print("model:", mid, " rule=Caesar  reps=%d sep=%r ntrain=%d ntest=%d seeds=%d (chance~0.038)"
      % (REPS, SEP, NTRAIN, NTEST, SEEDS))
for name, d in [("clean held-out (ceiling)", clean), ("no-transfer held-out", notrans),
                ("random held-out", rnd), ("FULL TRANSFER held-out", full),
                ("full transfer on TRAIN pairs", train_full)]:
    m, lo, hi = boot(d)
    print("  %-30s %.3f  [%.3f, %.3f]" % (name, m, lo, hi))
# primary test: transfer beats no-transfer on HELD-OUT
df = np.array(full) - np.array(notrans)
rng = np.random.default_rng(0)
bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
print("  TRANSFER-vs-notransfer (held-out): d=%.3f [%.3f, %.3f]  p=%.4f"
      % (df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
