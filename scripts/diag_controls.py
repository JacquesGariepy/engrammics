#!/usr/bin/env python3
"""Stronger, STRUCTURED negative controls than norm-matched random noise, for the
main associative-recall transfer. A shuffled-values engram has the SAME keys and
essentially the same singular spectrum as the real engram but encodes the wrong
associations; a wrong-skill engram is an unrelated skill's engram. If real
transfer beats both, the effect is specific to the engram's CONTENT, not its
structure or norm."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
REPS, NP, SEEDS = 3, 5, int(os.environ.get("SEEDS", 30))
AL, DG = list("ABCDEFGH"), [str(d) for d in range(10)]
PRIMER = "Q=4;Z=1;W=9;"
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


def demos(pairs): return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS
def engram(pairs): return getrec(fwd(ids(demos(pairs), bos=True)).past_key_values)


def recall(state, pairs):
    ok = 0
    for k, v in pairs:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s=" % k), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def skill(rng):
    return list(zip(rng.sample(AL, NP), rng.sample(DG, NP)))


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


R = {k: [] for k in ["full", "shuffled", "wrong"]}
for s in range(SEEDS):
    rng = random.Random(6000 + s)
    X = skill(rng); Y = skill(rng); W = skill(rng)
    keysX = [k for k, _ in X]; valsX = [v for _, v in X]
    shuf = valsX[:]
    while shuf == valsX:
        rng.shuffle(shuf)
    Xshuf = list(zip(keysX, shuf))                 # same keys, permuted values
    eX, eY = engram(X), engram(Y)
    eXshuf, eW = engram(Xshuf), engram(W)
    R["full"].append(recall(add(eY, eX), X))
    R["shuffled"].append(recall(add(eY, eXshuf), X))   # structured but wrong content
    R["wrong"].append(recall(add(eY, eW), X))          # unrelated skill engram

print("model:", mid, " structured controls  reps=%d pairs=%d seeds=%d (chance~0.1)" % (REPS, NP, SEEDS))
for name in ["full", "shuffled", "wrong"]:
    m, lo, hi = boot(R[name])
    print("  %-26s %.3f  [%.3f, %.3f]" % (name, m, lo, hi))
for nm, a, b in [("full - shuffled-values", R["full"], R["shuffled"]),
                 ("full - wrong-skill", R["full"], R["wrong"])]:
    df = np.array(a) - np.array(b); rng = np.random.default_rng(0)
    bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
    print("  %-24s d=%.3f [%.3f, %.3f] p=%.4f" %
          (nm, df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
