#!/usr/bin/env python3
"""Search for a task FORMAT and presentation under which this DeltaNet checkpoint
performs natural in-context associative recall well above chance. Without a real
ceiling, the transfer test is vacuous. All recall is NATURAL (full prompt, no
state injection) and uses add_special_tokens=False except a single leading BOS."""
import os, sys, random
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
BOS = tok.bos_token_id


def enc(text):
    ids = tok(text, add_special_tokens=False).input_ids
    return [BOS] + ids


def predict(text):
    ids = torch.tensor([enc(text)], device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    return tok.decode([int(out.logits[0, -1].argmax())]).strip()


def trial(make_prompt, make_query, n_pairs, alphabet, values, reps, seed):
    rng = random.Random(seed)
    keys = rng.sample(alphabet, n_pairs)
    vals = rng.sample(values, n_pairs)
    pairs = list(zip(keys, vals))
    body = make_prompt(pairs) * reps
    ok = 0
    for k, v in pairs:
        pred = predict(body + make_query(k))
        ok += (pred.strip() == v.strip())
    return ok / n_pairs


FORMATS = {
    "semicolon  k=v;":       (lambda P: "".join("%s=%s;" % (k, v) for k, v in P),
                              lambda k: "%s=" % k),
    "newline    k=v\\n":     (lambda P: "".join("%s=%s\n" % (k, v) for k, v in P),
                              lambda k: "%s=" % k),
    "arrow      k -> v\\n":  (lambda P: "".join("%s -> %s\n" % (k, v) for k, v in P),
                              lambda k: "%s ->" % k),
    "space      ' k v'":     (lambda P: "".join(" %s %s" % (k, v) for k, v in P),
                              lambda k: " %s" % k),
    "colon      'k: v\\n'":  (lambda P: "".join("%s: %s\n" % (k, v) for k, v in P),
                              lambda k: "%s:" % k),
}

ALPHA_LETTERS = list("ABCDEFGHIJKLMNOP")
ALPHA_WORDS = ["cat", "dog", "sun", "key", "box", "red", "owl", "map",
               "ice", "jet", "fox", "pen", "cup", "hat", "leaf", "star"]
DIGITS = [str(d) for d in range(10)]
WORDS_V = ["apple", "river", "stone", "cloud", "green", "north", "tiger",
           "music", "plant", "ocean", "light", "table", "happy", "zebra"]

SEEDS = list(range(8))


def avg(make_prompt, make_query, n_pairs, alphabet, values, reps):
    xs = [trial(make_prompt, make_query, n_pairs, alphabet, values, reps, 100 + s)
          for s in SEEDS]
    return sum(xs) / len(xs)


print("model:", mid)
print("\n# letters -> digits, 5 pairs, reps=1")
for name, (mp, mq) in FORMATS.items():
    print("  %-22s acc=%.2f" % (name, avg(mp, mq, 5, ALPHA_LETTERS, DIGITS, 1)))

print("\n# letters -> digits, 5 pairs, reps=3 (repeat the table 3x)")
for name, (mp, mq) in FORMATS.items():
    print("  %-22s acc=%.2f" % (name, avg(mp, mq, 5, ALPHA_LETTERS, DIGITS, 3)))

print("\n# word -> word, 5 pairs, reps=1 (arrow & colon & newline)")
for name in ["newline    k=v\\n", "arrow      k -> v\\n", "colon      'k: v\\n'"]:
    mp, mq = FORMATS[name]
    print("  %-22s acc=%.2f" % (name, avg(mp, mq, 5, ALPHA_WORDS, WORDS_V, 1)))

print("\n# letters -> digits, 3 pairs, reps=1")
for name, (mp, mq) in FORMATS.items():
    print("  %-22s acc=%.2f" % (name, avg(mp, mq, 3, ALPHA_LETTERS, DIGITS, 1)))

print("\n# letters -> digits, 3 pairs, reps=3")
for name, (mp, mq) in FORMATS.items():
    print("  %-22s acc=%.2f" % (name, avg(mp, mq, 3, ALPHA_LETTERS, DIGITS, 3)))
