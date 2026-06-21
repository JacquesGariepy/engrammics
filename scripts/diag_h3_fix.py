#!/usr/bin/env python3
"""Engineering targeted forgetting (H3) in the HARD regime (X and Y share the
alphabet, hence keys). Naive subtraction of X's engram from a joint X-then-Y state
removes X but damages Y. When X and Y share keys, removing the shared part of X
necessarily removes Y's use of those keys -- perfect forgetting with perfect
preservation is then impossible. The controllable remedy is the dual of the H2
fix: forget only X's keys that are ORTHOGONAL to Y's, P_{X\\Y} = proj((I-P_Y)U_X).
This keeps Y intact at the cost of leaving X's shared component recallable, i.e.
Y-safe (not complete) forgetting -- a tunable trade-off. We compare:
  (a) naive subtraction,  (b) full key-projection,  (c) X-only key-projection."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
SEEDS = int(os.environ.get("SEEDS", 24))
REPS, NP, HK = 3, 5, 128
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
def sub(a, b): return [x - y for x, y in zip(a, b)]


def demos(pairs): return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS


def key_basis(thresh=1e-2):
    """Per-layer per-head orthonormal key basis U (from the last captured keys).
    Returns list over layers of [H, dk, r_max] (zero-padded to a common r)."""
    Us = []
    for i in range(len(dn)):
        k = _cap[i].float().reshape(-1, NH, HK).transpose(0, 1)   # [H, seq, dk]
        _, s, Vh = torch.linalg.svd(k, full_matrices=False)         # Vh: [H, m, dk]
        keep = (s > thresh * s[:, :1])                              # [H, m]
        Us.append((Vh, keep))
    return Us


def engram_basis(pairs):
    out = fwd(ids(demos(pairs), bos=True))
    return getrec(out.past_key_values), key_basis()


def proj_full(Us):
    """P = U U^T over kept directions, per layer/head -> [1,H,dk,dk]."""
    P = []
    for Vh, keep in Us:
        mask = keep.to(Vh.dtype)
        P.append((Vh.transpose(-1, -2) @ (mask.unsqueeze(-1) * Vh)).unsqueeze(0))
    return P


def proj_x_minus_y(UX, UY):
    """Projector onto span((I - P_Y) U_X): X's key directions orthogonal to Y's."""
    P = []
    for (VhX, keepX), (VhY, keepY) in zip(UX, UY):
        Pl = torch.zeros(NH, HK, HK, device="cuda")
        for h in range(NH):
            ux = VhX[h][keepX[h]].transpose(0, 1)      # [dk, rX] basis of X keys
            uy = VhY[h][keepY[h]].transpose(0, 1)      # [dk, rY] basis of Y keys
            if ux.shape[1] == 0:
                continue
            if uy.shape[1] > 0:
                ux = ux - uy @ (uy.transpose(0, 1) @ ux)   # remove Y-component
            q, _ = torch.linalg.qr(ux)                      # orthonormalize residual
            # keep columns with non-trivial norm
            nz = (ux.norm(dim=0) > 1e-4)
            q = q[:, :int(nz.sum().item())] if nz.any() else q[:, :0]
            if q.shape[1]:
                Pl[h] = q @ q.transpose(0, 1)
        P.append(Pl.unsqueeze(0))
    return P


def apply_proj(state, P): return [(Ph @ s.float()).to(s.dtype) for s, Ph in zip(state, P)]
def remove(state, P): return [s - p for s, p in zip(state, apply_proj(state, P))]   # (I-P)state


def recall(state, pairs):
    ok = 0
    for k, v in pairs:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s=" % k), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def write_into(state, pairs):
    c = fwd(ids("", bos=True)).past_key_values
    setrec(c, state)
    return getrec(fwd(ids(demos(pairs), bos=False), cache=c).past_key_values)


def skill(rng): return list(zip(rng.sample(ALPHA, NP), rng.sample(DIG, NP)))

R = {k: [] for k in ["jX", "jY",
                     "naive_X", "naive_Y", "full_X", "full_Y", "xonly_X", "xonly_Y"]}
for s in range(SEEDS):
    rng = random.Random(9000 + s)
    X, Y = skill(rng), skill(rng)
    eX, UX = engram_basis(X)
    _,  UY = engram_basis(Y)
    P_X = proj_full(UX)
    P_XmY = proj_x_minus_y(UX, UY)
    joint = write_into(eX, Y)
    R["jX"].append(recall(joint, X)); R["jY"].append(recall(joint, Y))
    f_naive = sub(joint, eX)
    f_full = remove(joint, P_X)
    f_xonly = remove(joint, P_XmY)
    R["naive_X"].append(recall(f_naive, X)); R["naive_Y"].append(recall(f_naive, Y))
    R["full_X"].append(recall(f_full, X)); R["full_Y"].append(recall(f_full, Y))
    R["xonly_X"].append(recall(f_xonly, X)); R["xonly_Y"].append(recall(f_xonly, Y))
    del eX, UX, UY, P_X, P_XmY, joint, f_naive, f_full, f_xonly
    torch.cuda.empty_cache()


def m(x): return float(np.mean(x))
jX, jY = m(R["jX"]), m(R["jY"])
print("model:", mid, " seeds:", SEEDS, " HARD regime (shared alphabet)  chance=0.10")
print("  joint:           read X=%.3f  read Y=%.3f" % (jX, jY))
for nm, kx, ky in [("naive subtraction", "naive_X", "naive_Y"),
                   ("full key-projection", "full_X", "full_Y"),
                   ("X-only key-projection", "xonly_X", "xonly_Y")]:
    fx, fy = m(R[kx]), m(R[ky])
    print("  %-22s forget X->%.3f (dropX=%+.3f)   keep Y->%.3f (keepY=%+.3f)"
          % (nm, fx, jX - fx, fy, fy - jY))
