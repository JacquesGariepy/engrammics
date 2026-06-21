#!/usr/bin/env python3
"""A less-toy task: letter -> TWO-digit value (a multi-token output), to show the
engram carries multi-token associations, not only single-token lookups. The value
is scored by exact match of BOTH decoded tokens. Controls match the main protocol
and add a SHUFFLED-KEY engram (same keys and values, permuted pairing -- the dual
of the shuffled-values control): no-transfer, random, shuffled-key, full transfer."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
SEEDS = int(os.environ.get("SEEDS", 40))
REPS, NP = 3, 5
ALPHA = list("ABCDEFGH")
VALS = ["%02d" % v for v in range(10, 100)]          # two-digit strings (2 tokens)
PRIMER = "Q=40;Z=11;W=99;"
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


def recall_multi(state, pairs):
    """Greedily decode TWO tokens after 'k=' and require exact 2-token match."""
    ok = 0
    for k, v in pairs:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        gen = ""
        cache = c
        cur = ids("%s=" % k)
        for _ in range(2):
            out = fwd(cur, cache=cache)
            t = int(out.logits[0, -1].argmax())
            gen += tok.decode([t])
            cache = out.past_key_values
            cur = torch.tensor([[t]], device="cuda")
        ok += (gen.strip() == v)
    return ok / len(pairs)


def skill(seed):
    rng = random.Random(seed)
    return list(zip(rng.sample(ALPHA, NP), rng.sample(VALS, NP)))


R = {k: [] for k in ["clean", "notrans", "rnd", "shufkey", "full"]}
for s in range(SEEDS):
    rng = random.Random(70_000 + s)
    X = skill(70_000 + s)
    Y = skill(80_000 + s)
    keysX = [k for k, _ in X]; valsX = [v for _, v in X]
    sk = keysX[:]
    while sk == keysX:
        rng.shuffle(sk)
    Xshufkey = list(zip(sk, valsX))                  # permuted key<->value pairing
    eX, eY = engram(X), engram(Y)
    eXsk = engram(Xshufkey)
    R["clean"].append(recall_multi(eX, X))
    R["notrans"].append(recall_multi(eY, X))
    R["rnd"].append(recall_multi(add(eY, random_like(eX)), X))
    R["shufkey"].append(recall_multi(add(eY, eXsk), X))
    R["full"].append(recall_multi(add(eY, eX), X))


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("model:", mid, " task=letter->2-digit (multi-token)  reps=%d pairs=%d seeds=%d" % (REPS, NP, SEEDS))
print("  (a value is correct only if BOTH decoded tokens match; chance ~ 1/90 per pair)")
for nm, d in [("clean (ceiling)", "clean"), ("no-transfer", "notrans"),
              ("random engram", "rnd"), ("shuffled-key engram", "shufkey"),
              ("FULL TRANSFER", "full")]:
    m, lo, hi = boot(R[d])
    print("  %-24s %.3f  [%.3f, %.3f]" % (nm, m, lo, hi))
for nm, A, B in [("full - notransfer", "full", "notrans"),
                 ("full - random", "full", "rnd"),
                 ("full - shuffled-key", "full", "shufkey")]:
    df = np.array(R[A]) - np.array(R[B]); rng = np.random.default_rng(0)
    bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
    print("  %-22s d=%.3f [%.3f, %.3f] p=%.4f" %
          (nm, df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
