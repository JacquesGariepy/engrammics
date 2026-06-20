#!/usr/bin/env python3
"""Isolate the key-overlap variable cleanly: give X and Y FULLY disjoint token
sets AND disjoint separators, so the two skills share no key directions at all.
Compares three regimes of overlap and reports H2 (interference) and H3 (forget
keep-Y, by key-projection using captured DeltaNet keys):
  OVERLAP  : same alphabet, same '=' ';'
  SYMBOLS  : disjoint letters/digits, shared '=' ';'
  FULL     : disjoint letters/digits AND disjoint separators ('=' ';' vs ':' '|')
If interference falls to the noise floor and keep-Y becomes clean only in FULL,
the precondition (disjoint key subspaces) is demonstrated, not merely trended."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
REPS, NP = 3, 4
SEEDS = int(os.environ.get("SEEDS", 16))
HK = 128
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
def add(a, b): return [x + y for x, y in zip(a, b)]


class Skill:
    def __init__(self, pairs, sep, term, primer):
        self.pairs, self.sep, self.term, self.primer = pairs, sep, term, primer
    def demos(self): return "".join("%s%s%s%s" % (k, self.sep, v, self.term)
                                    for k, v in self.pairs) * REPS


def engram(sk):
    return getrec(fwd(ids(sk.demos(), bos=True)).past_key_values)


def keyproj(thresh=1e-2):
    P = []
    for i in range(len(dn)):
        k = _cap[i].float().reshape(-1, NH, HK).transpose(0, 1)
        _, s, Vh = torch.linalg.svd(k, full_matrices=False)
        mask = (s > thresh * s[:, :1]).to(Vh.dtype)
        P.append((Vh.transpose(-1, -2) @ (mask.unsqueeze(-1) * Vh)).unsqueeze(0))
    return P


def engram_keys(sk):
    out = fwd(ids(sk.demos(), bos=True))
    return getrec(out.past_key_values), keyproj()


def project(state, P): return [(Ph @ s.float()).to(s.dtype) for s, Ph in zip(state, P)]


def recall(state, sk):
    ok = 0
    for k, v in sk.pairs:
        c = fwd(ids(sk.primer, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s%s" % (k, sk.sep)), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == v
    return ok / len(sk.pairs)


def write_into(state, sk):
    c = fwd(ids("", bos=True)).past_key_values
    setrec(c, state)
    return getrec(fwd(ids(sk.demos(), bos=False), cache=c).past_key_values)


def mk(rng, keysrc, valsrc, sep, term, primer):
    return Skill(list(zip(rng.sample(keysrc, NP), rng.sample(valsrc, NP))), sep, term, primer)


D = list("0123456789")
SCEN = {
    "OVERLAP  (shared symbols & seps)": lambda r: (
        mk(r, list("ABCDEFGH"), D, "=", ";", "Q=4;Z=1;W=9;"),
        mk(r, list("ABCDEFGH"), D, "=", ";", "Q=4;Z=1;W=9;")),
    "SYMBOLS  (disjoint sym, shared seps)": lambda r: (
        mk(r, list("ABCD"), list("01234"), "=", ";", "Q=4;Z=1;W=9;"),
        mk(r, list("EFGH"), list("56789"), "=", ";", "Q=4;Z=1;W=9;")),
    "FULL     (disjoint sym AND seps)": lambda r: (
        mk(r, list("ABCD"), list("01234"), "=", ";", "Q=4;Z=1;W=9;"),
        mk(r, list("EFGH"), list("56789"), ":", "|", "R:5|T:6|Y:7|")),
}


def boot_mean(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("model:", mid, " seeds:", SEEDS, " pairs:", NP)
for name, gen in SCEN.items():
    H2, keepY, dropX = [], [], []
    for s in range(SEEDS):
        X, Y = gen(random.Random(5000 + s))
        eX, PX = engram_keys(X)
        eY = engram(Y)
        yb = recall(eY, Y); ya = recall(add(eY, eX), Y); H2.append(ya - yb)
        joint = write_into(eX, Y)
        jy = recall(joint, Y); jx = recall(joint, X)
        forg = sub(joint, project(joint, PX))
        keepY.append(recall(forg, Y) - jy); dropX.append(jx - recall(forg, X))
        del eX, PX, eY, joint, forg
        torch.cuda.empty_cache()
    m2, lo2, hi2 = boot_mean(H2)
    mk_, lok, hik = boot_mean(keepY)
    md, lod, hid = boot_mean(dropX)
    print("== %s ==" % name)
    print("   H2 dY            = %+.2f [%+.2f,%+.2f]" % (m2, lo2, hi2))
    print("   H3 keepY drop    = %+.2f [%+.2f,%+.2f]   dropX(proj)=%+.2f" % (mk_, lok, hik, md))
