#!/usr/bin/env python3
"""Can non-interference (H2) be ENGINEERED at injection time, without retraining?
The host holds skill Y. Naively adding X's engram damages Y because eX has mass in
Y's key directions. Fix: project the incoming engram onto the orthogonal complement
of the host's captured key subspace before adding it:  eX' = (I - P_Y) eX.
Then reading Y is untouched (eX' has no component in Y's keys) and X survives to the
extent its keys are disjoint from Y's. This is the governance projector repurposed
as a collision-avoidance allocator. We compare naive vs orthogonalized injection on
both Y-preservation (H2) and X-transfer (H1)."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
SEEDS = int(os.environ.get("SEEDS", 30))
REPS, NP = 3, 5
HK = 128
ALPHA, DIG = list("ABCDEFGH"), [str(d) for d in range(10)]
PRIMER = "Q=4;Z=1;W=9;"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id
dn = [m for _, m in model.named_modules() if m.__class__.__name__ == "DeltaNet"]
NH = dn[0].num_heads
_cap = {}
for i, m in enumerate(dn):
    m.k_conv1d.register_forward_hook((lambda j: lambda mod, inp, out: _cap.__setitem__(
        j, (out[0] if isinstance(out, tuple) else out).detach()))(i))


def ids(t, bos=False):
    x = tok(t, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + x if bos else x], device="cuda")


def getrec(c): return [c[i]["recurrent_state"].detach().clone() for i in range(len(c))]
def setrec(c, st):
    for i in range(len(c)): c[i]["recurrent_state"] = st[i]
def fwd(x, cache=None):
    _cap.clear()
    with torch.no_grad(): return model(input_ids=x, past_key_values=cache, use_cache=True)
def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]


def demos(pairs): return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS


def engram_keys(pairs):
    out = fwd(ids(demos(pairs), bos=True))
    rec = getrec(out.past_key_values)
    P = []
    for i in range(len(dn)):
        k = _cap[i].float().reshape(-1, NH, HK).transpose(0, 1)
        _, s, Vh = torch.linalg.svd(k, full_matrices=False)
        mask = (s > 1e-2 * s[:, :1]).to(Vh.dtype)
        P.append((Vh.transpose(-1, -2) @ (mask.unsqueeze(-1) * Vh)).unsqueeze(0))  # P_keys
    return rec, P


def project(state, P): return [(Ph @ s.float()).to(s.dtype) for s, Ph in zip(state, P)]
def ortho(state, P): return [s - p for s, p in zip(state, project(state, P))]   # (I-P)state


def recall(state, pairs):
    ok = 0
    for k, v in pairs:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s=" % k), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def skill(rng): return list(zip(rng.sample(ALPHA, NP), rng.sample(DIG, NP)))


R = {k: [] for k in ["Yb", "Ya_naive", "Ya_ortho", "X_naive", "X_ortho", "Xclean"]}
for s in range(SEEDS):
    rng = random.Random(8000 + s)
    X, Y = skill(rng), skill(rng)
    eX, _ = engram_keys(X)
    eY, PY = engram_keys(Y)               # PY = Y's key projector (the host)
    eX_ortho = ortho(eX, PY)              # (I - P_Y) eX  -> off Y's keys
    R["Xclean"].append(recall(eX, X))
    R["Yb"].append(recall(eY, Y))
    R["Ya_naive"].append(recall(add(eY, eX), Y))         # H2 failure
    R["Ya_ortho"].append(recall(add(eY, eX_ortho), Y))   # H2 fix?
    R["X_naive"].append(recall(add(eY, eX), X))          # H1 naive
    R["X_ortho"].append(recall(add(eY, eX_ortho), X))    # H1 with ortho


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("model:", mid, " seeds:", SEEDS, " (chance=0.10)")
for nm, d in [("X clean (ceiling)", R["Xclean"]),
              ("Y before", R["Yb"]),
              ("Y after, NAIVE add (H2 fail)", R["Ya_naive"]),
              ("Y after, ORTHO add (H2 fix)", R["Ya_ortho"]),
              ("X transfer, NAIVE", R["X_naive"]),
              ("X transfer, ORTHO", R["X_ortho"])]:
    m, lo, hi = boot(d)
    print("  %-32s %.3f  [%.3f, %.3f]" % (nm, m, lo, hi))
dY_n = np.mean(R["Ya_naive"]) - np.mean(R["Yb"])
dY_o = np.mean(R["Ya_ortho"]) - np.mean(R["Yb"])
print("  -> H2 host damage:  naive=%+.3f   ortho=%+.3f   (closer to 0 = fixed)" % (dY_n, dY_o))
print("  -> H1 transfer:     naive=%.3f    ortho=%.3f" % (np.mean(R["X_naive"]), np.mean(R["X_ortho"])))
