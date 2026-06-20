#!/usr/bin/env python3
"""Faithful key-subspace operations for the LM, using the ACTUAL DeltaNet keys
(captured from k_conv1d) instead of the engram's SVD. Tests:
  - Governance (H4): authorize X's key subspace -> admit recovers X; the
    orthogonal complement / a different party's keys -> blocked.
  - Forget-by-projection (H3): remove X's key subspace from a joint X-then-Y
    state -> drop X while keeping Y, vs forget-by-subtraction.
"""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id
REPS, NP = 3, 5
ALPHA = list("ABCDEFGH")
DIGITS = [str(d) for d in range(10)]
PRIMER = "Q=4;Z=1;W=9;"
HK = 128

dn_mods = [mod for _, mod in model.named_modules() if mod.__class__.__name__ == "DeltaNet"]
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


def demos(pairs):
    return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS


def getrec(cache):
    return [cache[i]["recurrent_state"].detach().clone() for i in range(len(cache))]


def setrec(cache, st):
    for i in range(len(cache)):
        cache[i]["recurrent_state"] = st[i]


def fwd(input_ids, cache=None):
    _cap.clear()
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return out


def skill(seed):
    rng = random.Random(seed)
    return list(zip(rng.sample(ALPHA, NP), rng.sample(DIGITS, NP)))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def key_projector(thresh=1e-2):
    """Build per-layer key projector P [1,NH,128,128] from the last captured keys.
    Call right after a fwd over a demos string."""
    P = []
    for i in range(len(dn_mods)):
        k = _cap[i].float()                         # [1, seq, NH*128]
        k = k.reshape(k.shape[1], NH, HK)           # [seq, NH, 128]
        Ph = torch.zeros(NH, HK, HK, device="cuda")
        for h in range(NH):
            kh = k[:, h, :]                          # [seq, 128]
            U, s, Vh = torch.linalg.svd(kh, full_matrices=False)
            r = int((s > thresh * s[0]).sum().item())
            Uk = Vh[:r].T                            # [128, r]
            Ph[h] = Uk @ Uk.T
        P.append(Ph.unsqueeze(0))               # keep float32
    return P


def project(state, P):
    return [(Ph @ s.float()).to(s.dtype) for s, Ph in zip(state, P)]


def engram_and_keys(pairs):
    out = fwd(ids(demos(pairs), bos=True))
    return getrec(out.past_key_values), key_projector()


def recall(state, pairs):
    ok = 0
    for k, v in pairs:
        carrier = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(carrier, state)
        o = fwd(ids("%s=" % k), cache=carrier)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def write_into(state, pairs):
    carrier = fwd(ids("", bos=True)).past_key_values
    setrec(carrier, state)
    c2 = fwd(ids(demos(pairs), bos=False), cache=carrier).past_key_values
    return getrec(c2)


seeds = list(range(8))
R = {k: [] for k in ["clean", "notrans", "full",
                     "gov_admit", "gov_deny_compl", "gov_deny_Ykeys",
                     "subF_X", "subF_Y", "projF_X", "projF_Y", "jX", "jY"]}
for s in seeds:
    X, Y = skill(1000 + s), skill(2000 + s)
    eX, PX = engram_and_keys(X)
    eY, PY = engram_and_keys(Y)
    R["clean"].append(recall(eX, X))
    R["notrans"].append(recall(eY, X))
    R["full"].append(recall(add(eY, eX), X))
    # governance via actual keys
    admit = project(eX, PX)                 # X through X's key region -> keeps X
    deny_c = sub(eX, admit)                  # orthogonal complement -> blocked
    deny_y = project(eX, PY)                 # X through Y's key region -> blocked
    R["gov_admit"].append(recall(add(eY, admit), X))
    R["gov_deny_compl"].append(recall(add(eY, deny_c), X))
    R["gov_deny_Ykeys"].append(recall(add(eY, deny_y), X))
    # forgetting from a joint X-then-Y state
    joint = write_into(eX, Y)
    R["jX"].append(recall(joint, X)); R["jY"].append(recall(joint, Y))
    subF = sub(joint, eX)                    # forget-by-subtraction (current)
    projF = sub(joint, project(joint, PX))   # forget-by-projection: (I-P_X) joint
    R["subF_X"].append(recall(subF, X)); R["subF_Y"].append(recall(subF, Y))
    R["projF_X"].append(recall(projF, X)); R["projF_Y"].append(recall(projF, Y))


def mean(x):
    return sum(x) / len(x)


print("model:", mid, " seeds:", len(seeds), " (chance=0.10)")
print("clean=%.2f  notransfer=%.2f  full=%.2f" % (mean(R["clean"]), mean(R["notrans"]), mean(R["full"])))
print("\nH4 governance (key-subspace):")
print("  admit (X thru X-keys)      = %.2f" % mean(R["gov_admit"]))
print("  deny  (orth. complement)   = %.2f" % mean(R["gov_deny_compl"]))
print("  deny  (X thru Y-keys)      = %.2f" % mean(R["gov_deny_Ykeys"]))
print("\nH3 forgetting (joint X-then-Y): jointX=%.2f jointY=%.2f" % (mean(R["jX"]), mean(R["jY"])))
print("  subtraction:  forget_X=%.2f forget_Y=%.2f  (keepY drop=%+.2f)"
      % (mean(R["subF_X"]), mean(R["subF_Y"]), mean(R["subF_Y"]) - mean(R["jY"])))
print("  projection :  forget_X=%.2f forget_Y=%.2f  (keepY drop=%+.2f)"
      % (mean(R["projF_X"]), mean(R["projF_Y"]), mean(R["projF_Y"]) - mean(R["jY"])))
