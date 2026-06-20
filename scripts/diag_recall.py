#!/usr/bin/env python3
"""Diagnose why the LM clean-write ceiling is low: separate (a) natural
in-context recall from (b) the inject-into-carrier mechanism, and check for a
spurious BOS in continuations."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()

print("add_bos_token:", getattr(tok, "add_bos_token", "?"),
      "bos:", tok.bos_token_id, "eos:", tok.eos_token_id)
print('tok("\\nA="):', tok("\nA=").input_ids)
print('tok("A="):', tok("A=").input_ids)
print('tok("3"):', tok("3").input_ids, "->", [tok.decode([t]) for t in tok("3").input_ids])
print('tok(";"):', tok(";").input_ids)


def demos(pairs):
    return "".join("%s=%s;" % (k, v) for k, v in pairs)


def argmax_tok(logits):
    return tok.decode([int(logits[0, -1].argmax())]).strip()


rng = random.Random(0)
keys = rng.sample(list("ABCDEFGH"), 5)
vals = rng.sample([str(d) for d in range(10)], 5)
pairs = list(zip(keys, vals))
D = demos(pairs)
print("\nDEMOS:", D, "pairs:", pairs)


def natural(sep):
    ok = 0
    rows = []
    for k, v in pairs:
        ids = tok(D + sep + k + "=", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        pred = argmax_tok(out.logits)
        ok += (pred == v)
        rows.append("%s=->%r(want %s)%s" % (k, pred, v, " OK" if pred == v else ""))
    print("  acc=%.2f  " % (ok / 5) + " ".join(rows))


print("\n--- NATURAL recall, sep='' (A= right after ;) ---")
natural("")
print("--- NATURAL recall, sep='\\n' ---")
natural("\n")

# ---- now the inject mechanism, mirroring the harness ----
def get(cache):
    return [cache[i]["recurrent_state"].detach().clone() for i in range(len(cache))]


def setst(cache, states):
    for i in range(len(cache)):
        cache[i]["recurrent_state"] = states[i]
    return cache


def cache_of(text):
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    return model(input_ids=ids, use_cache=True).past_key_values


def nxt(text, cache=None):
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return argmax_tok(out.logits), out.past_key_values


with torch.no_grad():
    eX = get(cache_of(D))
    print("\n--- INJECT recall, carrier from '\\n' (harness style) ---")
    ok = 0
    rows = []
    for k, v in pairs:
        _, carrier = nxt("\n")
        setst(carrier, eX)
        pred, _ = nxt(k + "=", cache=carrier)
        ok += (pred == v)
        rows.append("%s=->%r(want %s)%s" % (k, pred, v, " OK" if pred == v else ""))
    print("  acc=%.2f  " % (ok / 5) + " ".join(rows))

    # variant: carrier from BOS only (empty), inject, query without re-BOS
    print("--- INJECT recall, carrier from '' then query ';'+k+'=' ---")
    ok = 0
    rows = []
    for k, v in pairs:
        _, carrier = nxt(";")
        setst(carrier, eX)
        pred, _ = nxt(k + "=", cache=carrier)
        ok += (pred == v)
        rows.append("%s=->%r(want %s)%s" % (k, pred, v, " OK" if pred == v else ""))
    print("  acc=%.2f  " % (ok / 5) + " ".join(rows))
