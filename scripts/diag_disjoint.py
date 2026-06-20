#!/usr/bin/env python3
"""Test whether non-interference (H2) and clean forgetting (H3 keep-Y) recover
when the two skills occupy DISJOINT key subspaces, i.e. disjoint symbol sets.
The theory predicts clean superposition/forgetting only under subspace
disjointness; two LM skills that share vocabulary violate it. Compares
OVERLAPPING skills (both draw from ABCDEFGH x 0-9) to DISJOINT skills
(X: ABCD x 01234, Y: EFGH x 56789). Uses real-key forget-by-projection."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id
REPS, NP = 3, 4
PRIMER = "Q=4;Z=1;W=9;"
HK = 128
dn_mods = [m for _, m in model.named_modules() if m.__class__.__name__ == "DeltaNet"]
NH = dn_mods[0].num_heads
_cap = {}


def _hook(i):
    def h(mod, inp, out):
        _cap[i] = (out[0] if isinstance(out, tuple) else out).detach()
    return h


for i, mod in enumerate(dn_mods):
    mod.k_conv1d.register_forward_hook(_hook(i))


def ids(text, bos=False):
    t = tok(text, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + t if bos else t], device="cuda")


def demos(p):
    return "".join("%s=%s;" % (k, v) for k, v in p) * REPS


def getrec(c):
    return [c[i]["recurrent_state"].detach().clone() for i in range(len(c))]


def setrec(c, st):
    for i in range(len(c)):
        c[i]["recurrent_state"] = st[i]


def fwd(x, cache=None):
    _cap.clear()
    with torch.no_grad():
        return model(input_ids=x, past_key_values=cache, use_cache=True)


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def keyproj(thresh=1e-2):
    P = []
    for i in range(len(dn_mods)):
        k = _cap[i].float().reshape(_cap[i].shape[1], NH, HK)
        Ph = torch.zeros(NH, HK, HK, device="cuda")
        for h in range(NH):
            _, s, Vh = torch.linalg.svd(k[:, h, :], full_matrices=False)
            r = int((s > thresh * s[0]).sum().item())
            Ph[h] = Vh[:r].T @ Vh[:r]
        P.append(Ph.unsqueeze(0))
    return P


def project(state, P):
    return [(Ph @ s.float()).to(s.dtype) for s, Ph in zip(state, P)]


def engram(p):
    out = fwd(ids(demos(p), bos=True))
    return getrec(out.past_key_values), keyproj()


def recall(state, p):
    ok = 0
    for k, v in p:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s=" % k), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(p)


def write_into(state, p):
    c = fwd(ids("", bos=True)).past_key_values
    setrec(c, state)
    return getrec(fwd(ids(demos(p), bos=False), cache=c).past_key_values)


def make(seed, kx, vx, ky, vy):
    r = random.Random(seed)
    X = list(zip(r.sample(kx, NP), r.sample(vx, NP)))
    Y = list(zip(r.sample(ky, NP), r.sample(vy, NP)))
    return X, Y


def scenario(name, kx, vx, ky, vy, seeds):
    H2, keepY_sub, keepY_proj, dropX_proj, jY = [], [], [], [], []
    for s in seeds:
        X, Y = make(1000 + s, kx, vx, ky, vy)
        eX, PX = engram(X)
        eY, _ = engram(Y)
        yb = recall(eY, Y)
        ya = recall(add(eY, eX), Y)
        H2.append(ya - yb)
        joint = write_into(eX, Y)
        jy = recall(joint, Y); jY.append(jy)
        fsub = sub(joint, eX)
        fproj = sub(joint, project(joint, PX))
        keepY_sub.append(recall(fsub, Y) - jy)
        keepY_proj.append(recall(fproj, Y) - jy)
        dropX_proj.append(recall(joint, X) - recall(fproj, X))

    def m(x):
        return sum(x) / len(x)
    print("== %s ==" % name)
    print("  H2  dY (Y after-before adding X) = %+.2f" % m(H2))
    print("  H3  keepY drop  (subtraction)    = %+.2f" % m(keepY_sub))
    print("  H3  keepY drop  (projection)     = %+.2f   dropX(proj)=%+.2f"
          % (m(keepY_proj), m(dropX_proj)))


DIG = [str(d) for d in range(10)]
seeds = list(range(16))
print("model:", mid, " seeds:", len(seeds), " pairs:", NP)
scenario("OVERLAPPING  (X,Y both ABCDEFGH x 0-9)",
         list("ABCDEFGH"), DIG, list("ABCDEFGH"), DIG, seeds)
scenario("DISJOINT     (X: ABCD x 01234 | Y: EFGH x 56789)",
         list("ABCD"), list("01234"), list("EFGH"), list("56789"), seeds)
