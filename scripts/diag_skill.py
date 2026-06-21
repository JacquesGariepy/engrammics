#!/usr/bin/env python3
"""Can the base DeltaNet learn a GENERALIZING rule in-context (not a lookup
table)? We teach a Caesar shift f(x)=x+k on a TRAIN set of letters and test on
HELD-OUT letters. High held-out accuracy = the rule generalizes; this is the
precondition for a real (non-dictionary) skill-transfer test. We also report a
wrong-rule control (demos of a different shift) on the same held-out letters."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True).to("cuda").eval()
BOS = tok.bos_token_id
AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ids(text, bos=True):
    t = tok(text, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + t if bos else t], device="cuda")


def predict(prompt):
    with torch.no_grad():
        out = model(input_ids=ids(prompt), use_cache=False)
    return tok.decode([int(out.logits[0, -1].argmax())]).strip()


def caesar(x, k):
    return AL[(AL.index(x) + k) % 26]


def demos(pairs, sep, reps):
    return "".join("%s%s%s;" % (k, sep, v) for k, v in pairs) * reps


def trial(k, n_train, n_test, sep, reps, seed):
    rng = random.Random(seed)
    letters = rng.sample(list(AL), n_train + n_test)
    train = letters[:n_train]
    test = letters[n_train:]
    body = demos([(x, caesar(x, k)) for x in train], sep, reps)
    # wrong-rule control: same train letters, a DIFFERENT shift kk
    kk = (k + rng.randint(1, 9)) % 26
    if kk == 0:
        kk = 1
    body_wrong = demos([(x, caesar(x, kk)) for x in train], sep, reps)
    ok = okw = 0
    for x in test:
        want = caesar(x, k)
        ok += (predict(body + "%s%s" % (x, sep)) == want)
        okw += (predict(body_wrong + "%s%s" % (x, sep)) == want)
    return ok / n_test, okw / n_test


def avg(k, n_train, n_test, sep, reps, seeds):
    rs = [trial(k, n_train, n_test, sep, reps, 700 + s) for s in seeds]
    held = sum(r[0] for r in rs) / len(rs)
    wrong = sum(r[1] for r in rs) / len(rs)
    return held, wrong


seeds = list(range(10))
print("model:", mid, "  (chance ~ 1/26 = 0.038)")
print("\n# Caesar shift, held-out generalization. cols: held-out acc | wrong-rule acc")
for sep in [">", "="]:
    for reps in [1, 3, 6]:
        for k in [1, 2, 3, 5]:
            held, wrong = avg(k, 6, 6, sep, reps, seeds)
            print("  sep=%r reps=%d k=%d :  held=%.2f  wrong=%.2f" % (sep, reps, k, held, wrong))
