#!/usr/bin/env python3
"""Test the DELTA-engram fix against the current FULL-state engram on all five
hypotheses. The current LM engram is the full recurrent state after [BOS]+demos,
so summing two engrams double-counts the [BOS] baseline (2b + dX + dY); this
corrupts superposition and is the suspected cause of the H2/H3/H4 degradation.
Fix B: engram = state([BOS]+demos) - state([BOS]) (a pure delta, as in the toy),
and read back as baseline + (sum of deltas), so the baseline appears exactly once.
"""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id
REPS = 3
NP = 5
ALPHA = list("ABCDEFGH")
DIGITS = [str(d) for d in range(10)]
PRIMER = "Q=4;Z=1;W=9;"


def ids(text, bos=False):
    t = tok(text, add_special_tokens=False).input_ids
    if bos:
        t = [BOS] + t
    return torch.tensor([t], device="cuda")


def demos(pairs):
    return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS


def get(cache):
    return [cache[i]["recurrent_state"].detach().clone() for i in range(len(cache))]


def setrec(cache, states):
    for i in range(len(cache)):
        cache[i]["recurrent_state"] = states[i]


def fwd(input_ids, cache=None):
    with torch.no_grad():
        return model(input_ids=input_ids, past_key_values=cache, use_cache=True)


def skill(seed):
    rng = random.Random(seed)
    return list(zip(rng.sample(ALPHA, NP), rng.sample(DIGITS, NP)))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


# baseline b0 = recurrent state after [BOS]
B0 = get(fwd(ids("", bos=True)).past_key_values)

# ---------- formulation A: full-state engram (current harness) ----------
def engram_A(pairs):
    return get(fwd(ids(demos(pairs), bos=True)).past_key_values)


def recall_A(state, pairs):
    ok = 0
    for k, v in pairs:
        carrier = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(carrier, state)
        out = fwd(ids("%s=" % k), cache=carrier)
        ok += tok.decode([int(out.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def write_into_A(state, pairs):
    carrier = fwd(ids("", bos=True)).past_key_values
    setrec(carrier, state)
    c2 = fwd(ids(demos(pairs), bos=False), cache=carrier).past_key_values
    return get(c2)


# ---------- formulation B: delta engram + single baseline ----------
def engram_B(pairs):
    full = get(fwd(ids(demos(pairs), bos=True)).past_key_values)
    return sub(full, B0)                       # pure delta


def recall_B(delta, pairs):
    inj = add(B0, delta)                        # baseline once + summed deltas
    ok = 0
    for k, v in pairs:
        carrier = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(carrier, inj)
        out = fwd(ids("%s=" % k), cache=carrier)
        ok += tok.decode([int(out.logits[0, -1].argmax())]).strip() == v
    return ok / len(pairs)


def write_into_B(delta, pairs):
    # inject X (baseline+delta), continue writing Y, return joint delta
    carrier = fwd(ids("", bos=True)).past_key_values
    setrec(carrier, add(B0, delta))
    c2 = fwd(ids(demos(pairs), bos=False), cache=carrier).past_key_values
    return sub(get(c2), B0)


def key_subspace(delta, r):
    out = []
    for s in delta:
        U = torch.linalg.svd(s.float(), full_matrices=False)[0]
        out.append(U[..., :min(r, U.shape[-1])])
    return out


def project(delta, U):
    return [(Ui @ (Ui.transpose(-1, -2) @ s.float())).to(s.dtype) for s, Ui in zip(delta, U)]


def run(engram, recall, write_into, seeds):
    res = {k: [] for k in ["clean", "notransfer", "transfer", "Yb", "Ya",
                            "jX", "jY", "fX", "fY", "admit", "deny"]}
    for s in seeds:
        X, Y = skill(1000 + s), skill(2000 + s)
        eX, eY = engram(X), engram(Y)
        res["clean"].append(recall(eX, X))
        res["notransfer"].append(recall(eY, X))
        res["transfer"].append(recall(add(eY, eX), X))
        res["Yb"].append(recall(eY, Y))
        res["Ya"].append(recall(add(eY, eX), Y))
        joint = write_into(eX, Y)
        forget = sub(joint, eX)
        res["jX"].append(recall(joint, X)); res["jY"].append(recall(joint, Y))
        res["fX"].append(recall(forget, X)); res["fY"].append(recall(forget, Y))
        U = key_subspace(eX, NP)
        admit = project(eX, U)
        res["admit"].append(recall(add(eY, admit), X))
        res["deny"].append(recall(add(eY, sub(eX, admit)), X))
    return {k: sum(v) / len(v) for k, v in res.items()}


seeds = list(range(10))
print("model:", mid, " seeds:", len(seeds), " (chance=0.10)")
for tag, (e, r, w) in {
    "A full-state (current)": (engram_A, recall_A, write_into_A),
    "B delta + 1 baseline ": (engram_B, recall_B, write_into_B),
}.items():
    m = run(e, r, w, seeds)
    print("\n== %s ==" % tag)
    print("  clean=%.2f  notransfer=%.2f  transfer=%.2f   (H1)" % (m["clean"], m["notransfer"], m["transfer"]))
    print("  Y_before=%.2f  Y_after=%.2f   dY=%+.2f (H2)" % (m["Yb"], m["Ya"], m["Ya"] - m["Yb"]))
    print("  joint_X=%.2f joint_Y=%.2f  forget_X=%.2f forget_Y=%.2f  (H3: dropX=%+.2f keepY=%+.2f)"
          % (m["jX"], m["jY"], m["fX"], m["fY"], m["jX"] - m["fX"], m["fY"] - m["jY"]))
    print("  admit=%.2f deny=%.2f   (H4: admit-deny=%+.2f)" % (m["admit"], m["deny"], m["admit"] - m["deny"]))
