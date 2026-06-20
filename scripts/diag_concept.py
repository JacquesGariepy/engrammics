#!/usr/bin/env python3
"""Can the model apply a KNOWN-CONCEPT generalizing rule in context (unlike an
arbitrary Caesar shift)? We test two functions over letters that align with
pretrained structure: vowel/consonant, and first-half/second-half of alphabet.
A train set teaches the mapping (letter -> class digit); we score HELD-OUT
letters. A wrong-rule control swaps the class labels. If held-out >> wrong-rule
and >> chance, the model generalizes a real function -> rule transfer is testable."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id
AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOWELS = set("AEIOU")


def ids(t, bos=True):
    x = tok(t, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + x if bos else x], device="cuda")


def predict(p):
    with torch.no_grad():
        out = model(input_ids=ids(p), use_cache=False)
    return tok.decode([int(out.logits[0, -1].argmax())]).strip()


def cls_vowel(x, a, b):
    return a if x in VOWELS else b


def cls_half(x, a, b):
    return a if AL.index(x) < 13 else b


def demos(pairs, sep, reps):
    return "".join("%s%s%s;" % (k, v, "") for k, v in [(k, s) for k, s in pairs]).replace("", "") if False else \
        "".join("%s%s%s;" % (k, sep, v) for k, v in pairs) * reps


def trial(fn, sep, reps, seed):
    rng = random.Random(seed)
    a, b = rng.sample([str(d) for d in range(10)], 2)
    letters = rng.sample(list(AL), 16)
    train, test = letters[:8], letters[8:]
    # ensure both classes present in train
    body = demos([(x, fn(x, a, b)) for x in train], sep, reps)
    body_w = demos([(x, fn(x, b, a)) for x in train], sep, reps)   # swapped labels
    ok = okw = 0
    for x in test:
        want = fn(x, a, b)
        ok += predict(body + "%s%s" % (x, sep)) == want
        okw += predict(body_w + "%s%s" % (x, sep)) == want
    return ok / len(test), okw / len(test)


def avg(fn, sep, reps, seeds):
    rs = [trial(fn, sep, reps, 800 + s) for s in seeds]
    return sum(r[0] for r in rs) / len(rs), sum(r[1] for r in rs) / len(rs)


seeds = list(range(12))
print("model:", mid, "  (chance=0.5 for 2-class; held vs wrong-label)")
for name, fn in [("vowel/consonant", cls_vowel), ("first/second half", cls_half)]:
    print("# %s" % name)
    for sep in ["=", ">"]:
        for reps in [1, 3]:
            h, w = avg(fn, sep, reps, seeds)
            print("  sep=%r reps=%d :  held=%.2f  wrong-label=%.2f" % (sep, reps, h, w))
