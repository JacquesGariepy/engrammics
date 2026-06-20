#!/usr/bin/env python3
"""Find a skill-agnostic carrier primer so that recurrent-state-only injection
(the only kind compatible with additive transfer) recovers the natural recall
ceiling. The primer supplies a realistic conv-window; the recurrent_state is then
overwritten by the engram, so the primer leaks no skill information."""
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


def setrec(cache, states):
    for i in range(len(cache)):
        cache[i]["recurrent_state"] = states[i]


def fwd(input_ids, cache=None):
    with torch.no_grad():
        return model(input_ids=input_ids, past_key_values=cache, use_cache=True)


def skill(seed):
    rng = random.Random(seed)
    return list(zip(rng.sample(ALPHA, NP), rng.sample(DIGITS, NP)))


def engram(pairs):
    return get(fwd(ids(demos(pairs), bos=True)).past_key_values)


def recall_primed(state, pairs, primer, qprefix):
    ok = 0
    for k, v in pairs:
        carrier = fwd(ids(primer, bos=True)).past_key_values
        setrec(carrier, state)
        out = fwd(ids(qprefix + "%s=" % k), cache=carrier)
        pred = tok.decode([int(out.logits[0, -1].argmax())]).strip()
        ok += (pred == v)
    return ok / len(pairs)


def add(a, b):
    return [x + y for x, y in zip(a, b)]


# (primer, query-prefix) candidates. Primer is processed (with BOS) to seed the
# conv window; recurrent_state is then overwritten, so primer content is inert.
VARIANTS = {
    "BOS only,    q='k='":      ("",        ""),
    "primer ';',  q='k='":      (";",       ""),
    "primer '0=0;', q='k='":    ("0=0;",    ""),
    "primer 'Q=4;', q='k='":    ("Q=4;",    ""),
    "BOS only,    q=';k='":     ("",        ";"),
    "primer 'Q=4;Z=1;', q='k='": ("Q=4;Z=1;", ""),
    "primer 'Q=4;Z=1;W=9;', q='k='": ("Q=4;Z=1;W=9;", ""),
}

seeds = list(range(8))
print("model:", mid, " reps:", REPS, " pairs:", NP, " seeds:", len(seeds))
print("\n# single-engram recurrent-only recall (target: ~1.00)")
best = None
for name, (pr, qp) in VARIANTS.items():
    accs = [recall_primed(engram(skill(1000 + s)), skill(1000 + s), pr, qp) for s in seeds]
    a = sum(accs) / len(accs)
    print("  %-32s acc=%.3f" % (name, a))

print("\n# transfer (eY+eX read X) and non-interference (read Y) with best primers")
for name in ["primer 'Q=4;Z=1;W=9;', q='k='", "primer 'Q=4;', q='k='", "primer ';',  q='k='"]:
    pr, qp = VARIANTS[name]
    tx, ny, by = [], [], []
    for s in seeds:
        X, Y = skill(1000 + s), skill(2000 + s)
        eX, eY = engram(X), engram(Y)
        sup = add(eY, eX)
        by.append(recall_primed(eY, Y, pr, qp))
        tx.append(recall_primed(sup, X, pr, qp))
        ny.append(recall_primed(sup, Y, pr, qp))
    print("  %-32s baseY=%.3f  transferX=%.3f  keepY=%.3f"
          % (name, sum(by) / len(by), sum(tx) / len(tx), sum(ny) / len(ny)))
