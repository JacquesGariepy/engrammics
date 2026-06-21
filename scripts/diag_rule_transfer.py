#!/usr/bin/env python3
"""THE decisive test: does a transferred engram carry a GENERALIZING RULE?
The skill is a vowel/consonant classifier with a per-seed label assignment
(vowel->a, consonant->b). The base model applies this concept to HELD-OUT letters
(verified: held ~0.76 vs wrong-label ~0.24). We write the rule's engram from a
TRAIN set of letters, inject it (recurrent state only, neutral primer) into a
recipient holding an unrelated constant-output skill, and test on HELD-OUT letters
never demonstrated. If full transfer beats the no-transfer and random controls on
held-out letters, the engram transferred a RULE that generalizes -- skill, not
dictionary. The wrong-label control (swap a<->b) tests label specificity."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
REPS = int(os.environ.get("REPS", 3))
NTRAIN, NTEST = int(os.environ.get("NTRAIN", 8)), int(os.environ.get("NTEST", 8))
SEEDS = int(os.environ.get("SEEDS", 40))
AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOW = set("AEIOU")
PRIMER = "1=1;2=2;3=3;"
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True).to("cuda").eval()
BOS = tok.bos_token_id


def ids(t, bos=False):
    x = tok(t, add_special_tokens=False).input_ids
    return torch.tensor([[BOS] + x if bos else x], device="cuda")


def getrec(c): return [c[i]["recurrent_state"].detach().clone() for i in range(len(c))]
def setrec(c, st):
    for i in range(len(c)): c[i]["recurrent_state"] = st[i]
def fwd(x, cache=None):
    with torch.no_grad(): return model(input_ids=x, past_key_values=cache, use_cache=True)
def add(a, b): return [x + y for x, y in zip(a, b)]


def label(x, a, b): return a if x in VOW else b


def demos_rule(letters, a, b):
    return "".join("%s=%s;" % (x, label(x, a, b)) for x in letters) * REPS


def demos_const(letters, c):
    return "".join("%s=%s;" % (x, c) for x in letters) * REPS


def engram(text): return getrec(fwd(ids(text, bos=True)).past_key_values)


def random_like(e, r=16):
    out = []
    for s in e:
        k = min(r, s.shape[-1]); g = torch.Generator(device=s.device)
        g.manual_seed(int(s.float().abs().sum().item() * 1e3) % (2 ** 31))
        A = torch.randn(*s.shape[:-1], k, device=s.device, dtype=torch.float32, generator=g)
        B = torch.randn(*s.shape[:-2], k, s.shape[-1], device=s.device, dtype=torch.float32, generator=g)
        M = A @ B
        out.append((M * (s.float().norm() / (M.norm() + 1e-9))).to(s.dtype))
    return out


def score(state, test_letters, a, b):
    ok = 0
    for x in test_letters:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s=" % x), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == label(x, a, b)
    return ok / len(test_letters)


clean, notrans, rnd, full, wrong = [], [], [], [], []
for s in range(SEEDS):
    rng = random.Random(4000 + s)
    a, b = rng.sample([str(d) for d in range(10)], 2)
    cY = rng.choice([d for d in "0123456789" if d not in (a, b)])
    L = rng.sample(list(AL), NTRAIN + NTEST)
    train, test = L[:NTRAIN], L[NTRAIN:]
    # guarantee both classes appear in train
    if all(x in VOW for x in train) or all(x not in VOW for x in train):
        continue
    trainY = rng.sample(list(AL), NTRAIN)
    try:
        eX = engram(demos_rule(train, a, b))       # the rule engram
        eXw = engram(demos_rule(train, b, a))      # swapped labels
        eY = engram(demos_const(trainY, cY))       # recipient's unrelated skill
        # compute all conditions before appending so the lists stay aligned
        c = score(eX, test, a, b)                  # rule generalizes? (ceiling)
        nt = score(eY, test, a, b)                 # recipient alone
        rn = score(add(eY, random_like(eX)), test, a, b)
        fl = score(add(eY, eX), test, a, b)        # RULE TRANSFER on held-out
        wr = score(add(eY, eXw), test, a, b)       # swapped-label engram
    except Exception as e:
        sys.stderr.write("skip seed %d: %r\n" % (s, e))
        continue
    clean.append(c); notrans.append(nt); rnd.append(rn)
    full.append(fl); wrong.append(wr)


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("model:", mid, " rule=vowel/consonant  reps=%d ntrain=%d ntest=%d seeds=%d"
      % (REPS, NTRAIN, NTEST, len(clean)))
for nm, d in [("clean held-out (rule ceiling)", clean), ("no-transfer held-out", notrans),
              ("random held-out", rnd), ("wrong-label engram held-out", wrong),
              ("FULL RULE TRANSFER held-out", full)]:
    m, lo, hi = boot(d)
    print("  %-32s %.3f  [%.3f, %.3f]" % (nm, m, lo, hi))
for nm, A, B in [("full - notransfer", full, notrans), ("full - random", full, rnd),
                 ("full - wrong-label", full, wrong)]:
    df = np.array(A) - np.array(B); rng = np.random.default_rng(0)
    bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
    print("  %-22s d=%.3f [%.3f, %.3f] p=%.4f" %
          (nm, df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
