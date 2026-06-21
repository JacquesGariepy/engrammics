#!/usr/bin/env python3
"""The motivating scenario: transfer an engram between TWO DIFFERENT checkpoints
(not two copies of one model). Donor A = delta_net-1.3B-100B, recipient
B = delta_net-1.3B-8K-100B (same architecture and dims, different training run, so
the recurrent states are dimensionally compatible but addressed in different key
bases). Naive cross-checkpoint injection should fail; a learned linear alignment
of the recurrent state should recover it.

Alignment: if B's keys equal A's keys mapped by W (k_B = W k_A), then the engram
F = sum_i k_i v_i^T satisfies F_B = W F_A. We learn a per-layer, per-head W by
least squares F_B ~= W F_A over a set of ANCHOR skills, then transfer W . eX_A
into B and read. Reported against B's own engram (upper bound) and naive transfer."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

A_ID = os.environ.get("A_ID", os.path.expanduser("~/models/delta_net-1.3B-100B"))
B_ID = os.environ.get("B_ID", os.path.expanduser("~/models/delta_net-1.3B-8K-100B"))
N_ANCHOR = int(os.environ.get("ANCHORS", 24))
SEEDS = int(os.environ.get("SEEDS", 25))
REPS, NP = 3, 5
ALPHA, DIG = list("ABCDEFGH"), [str(d) for d in range(10)]
PRIMER = "Q=4;Z=1;W=9;"

tok = AutoTokenizer.from_pretrained(A_ID, trust_remote_code=True)
BOS = tok.bos_token_id
print("loading A and B...", flush=True)
mA = AutoModelForCausalLM.from_pretrained(A_ID, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True).to("cuda").eval()
mB = AutoModelForCausalLM.from_pretrained(B_ID, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True).to("cuda").eval()


def ids(t, bos=False):
    x = tok(t, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + x if bos else x], device="cuda")


def getrec(c): return [c[i]["recurrent_state"].detach().clone() for i in range(len(c))]
def setrec(c, st):
    for i in range(len(c)): c[i]["recurrent_state"] = st[i]
def fwd(model, x, cache=None):
    with torch.no_grad(): return model(input_ids=x, past_key_values=cache, use_cache=True)


def demos(pairs): return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS
def engram(model, pairs): return getrec(fwd(model, ids(demos(pairs), bos=True)).past_key_values)


def recall(model, state, pairs):
    ok = 0
    for k, v in pairs:
        c = fwd(model, ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(model, ids("%s=" % k), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def skill(seed):
    rng = random.Random(seed)
    return list(zip(rng.sample(ALPHA, NP), rng.sample(DIG, NP)))


# ---- learn per-layer per-head alignment W: F_B ~= W F_A over anchors ----
# state[l]: [1, H, dk, dv]; for head h, F_A[h], F_B[h] are [dk, dv]. Treat each
# column as an example: W[h] @ F_A[h][:,j] = F_B[h][:,j]. Stack columns over anchors.
print("learning alignment from %d anchors..." % N_ANCHOR, flush=True)
anchorsA, anchorsB = [], []
for a in range(N_ANCHOR):
    sk = skill(50_000 + a)
    anchorsA.append(engram(mA, sk)); anchorsB.append(engram(mB, sk))

L = len(anchorsA[0]); H = anchorsA[0][0].shape[1]; DK = anchorsA[0][0].shape[2]
W = []  # W[l]: [H, dk, dk]
for l in range(L):
    Wl = torch.zeros(H, DK, DK, device="cuda")
    for h in range(H):
        XA = torch.cat([anchorsA[a][l][0, h].float() for a in range(N_ANCHOR)], dim=1)  # [dk, dv*A]
        XB = torch.cat([anchorsB[a][l][0, h].float() for a in range(N_ANCHOR)], dim=1)  # [dk, dv*A]
        # solve W XA = XB  ->  W = XB XA^T (XA XA^T + eps I)^-1
        G = XA @ XA.transpose(-1, -2)
        G = G + 1e-2 * torch.eye(DK, device="cuda") * G.diagonal().mean()
        Wl[h] = (XB @ XA.transpose(-1, -2)) @ torch.linalg.inv(G)
    W.append(Wl)


def align(state):
    out = []
    for l, s in enumerate(state):
        out.append(torch.einsum("hij,bhjk->bhik", W[l], s.float()).to(s.dtype))
    return out


R = {k: [] for k in ["A_ceiling", "B_ceiling", "naive", "aligned"]}
for s in range(SEEDS):
    X = skill(60_000 + s)
    eA = engram(mA, X); eB = engram(mB, X)
    R["A_ceiling"].append(recall(mA, eA, X))          # donor reads its own engram
    R["B_ceiling"].append(recall(mB, eB, X))          # recipient's own engram (upper bound)
    R["naive"].append(recall(mB, eA, X))              # A's engram into B, no alignment
    R["aligned"].append(recall(mB, align(eA), X))     # A's engram aligned into B


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("\nCROSS-CHECKPOINT: A=%s  B=%s" % (os.path.basename(A_ID), os.path.basename(B_ID)))
print("anchors=%d seeds=%d (chance=0.10)" % (N_ANCHOR, SEEDS))
for nm, d in [("A reads own engram (ref)", R["A_ceiling"]),
              ("B reads own engram (upper bound)", R["B_ceiling"]),
              ("naive A->B (no alignment)", R["naive"]),
              ("aligned A->B (learned W)", R["aligned"])]:
    m, lo, hi = boot(d)
    print("  %-34s %.3f  [%.3f, %.3f]" % (nm, m, lo, hi))
df = np.array(R["aligned"]) - np.array(R["naive"]); rng = np.random.default_rng(0)
bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
print("  aligned - naive   d=%.3f [%.3f, %.3f] p=%.4f" %
      (df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
