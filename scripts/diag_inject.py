#!/usr/bin/env python3
"""On the winning format (semicolon, reps=3, 5 pairs), verify that injecting the
engram (the recurrent_state) into a neutral carrier reproduces the natural recall
ceiling, and that additive superposition of two engrams keeps both readable.
This validates the faithful fast-weight-transfer mechanism the harness needs."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id
REPS = 3
NP = 5
ALPHA = list("ABCDEFGHIJKLMNOP")
DIGITS = [str(d) for d in range(10)]


def ids(text, bos=False):
    t = tok(text, add_special_tokens=False).input_ids
    if bos:
        t = [BOS] + t
    return torch.tensor([t], device="cuda")


def demos(pairs):
    return "".join("%s=%s;" % (k, v) for k, v in pairs) * REPS


def get(cache):
    return [cache[i]["recurrent_state"].detach().clone() for i in range(len(cache))]


def getconv(cache):
    return [cache[i]["conv_state"] for i in range(len(cache))]


def setrec(cache, states):
    for i in range(len(cache)):
        cache[i]["recurrent_state"] = states[i]


def setconv(cache, convs):
    for i in range(len(cache)):
        cache[i]["conv_state"] = convs[i]


def fwd(input_ids, cache=None):
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return out


def skill(seed):
    rng = random.Random(seed)
    keys = rng.sample(ALPHA, NP)
    vals = rng.sample(DIGITS, NP)
    return list(zip(keys, vals))


def engram(pairs):
    return get(fwd(ids(demos(pairs), bos=True)).past_key_values)


def engram_full(pairs):
    c = fwd(ids(demos(pairs), bos=True)).past_key_values
    return get(c), getconv(c)


def recall_natural(pairs):
    body = demos(pairs)
    ok = 0
    for k, v in pairs:
        out = fwd(ids(body + "%s=" % k, bos=True))
        pred = tok.decode([int(out.logits[0, -1].argmax())]).strip()
        ok += (pred == v)
    return ok / len(pairs)


def recall_inject(state, pairs, conv=None):
    ok = 0
    for k, v in pairs:
        carrier = fwd(ids("", bos=True)).past_key_values  # process [BOS]
        setrec(carrier, state)
        if conv is not None:
            setconv(carrier, conv)
        out = fwd(ids("%s=" % k), cache=carrier)
        pred = tok.decode([int(out.logits[0, -1].argmax())]).strip()
        ok += (pred == v)
    return ok / len(pairs)


def add(a, b):
    return [x + y for x, y in zip(a, b)]


seeds = list(range(8))
nat, inj, injc, sup_x, sup_y, base_y = [], [], [], [], [], []
for s in seeds:
    X = skill(1000 + s)
    Y = skill(2000 + s)
    eX = engram(X)
    eXf, convX = engram_full(X)
    eY = engram(Y)
    nat.append(recall_natural(X))
    inj.append(recall_inject(eX, X))
    injc.append(recall_inject(eXf, X, conv=convX))
    base_y.append(recall_inject(eY, Y))          # B holds Y, read Y
    sup = add(eY, eX)                              # superpose
    sup_x.append(recall_inject(sup, X))           # read X from B+X
    sup_y.append(recall_inject(sup, Y))           # read Y from B+X


def m(x):
    return sum(x) / len(x)


print("model:", mid, " reps:", REPS, " pairs:", NP, " seeds:", len(seeds))
print("natural recall X (ceiling)          : %.3f" % m(nat))
print("inject eX (recurrent only), read X  : %.3f" % m(inj))
print("inject eX + conv_state,    read X   : %.3f" % m(injc))
print("inject eY, read Y (B's own skill)   : %.3f" % m(base_y))
print("inject eY+eX, read X (TRANSFER)     : %.3f" % m(sup_x))
print("inject eY+eX, read Y (non-interf.)  : %.3f" % m(sup_y))
