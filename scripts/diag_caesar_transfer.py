#!/usr/bin/env python3
"""Transfer of an ARBITRARY symbolic rule (Caesar shift f(x)=x+k, k=2 default)
on a model that can learn it in context (RWKV-7-2.9B). The rule is NOT a
pretrained concept; the model induces it from train demonstrations and applies it
to HELD-OUT letters (verified ~0.33 in-context for k=2). We write the rule's
engram from a train set, inject it (recurrent state only) into a recipient, and
score held-out letters never demonstrated. If the engram alone (neutral recipient)
reproduces the in-context rule level while a wrong-shift engram and a random
engram do not, the transferred fast state carries an arbitrary generalizing rule.

Env: K (shift, default 2), REPS (3), NTRAIN (10), NTEST (8), SEEDS (50)."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
K = int(os.environ.get("K", 2))
REPS = int(os.environ.get("REPS", 3))
NTRAIN, NTEST = int(os.environ.get("NTRAIN", 10)), int(os.environ.get("NTEST", 8))
SEEDS = int(os.environ.get("SEEDS", 50))
AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEP = ">"
PRIMER = "1>1;2>2;3>3;"
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


def caesar(x, k): return AL[(AL.index(x) + k) % 26]


def demos(letters, k):
    return "".join("%s%s%s;" % (x, SEP, caesar(x, k)) for x in letters) * REPS


def engram(text): return getrec(fwd(ids(text, bos=True)).past_key_values)


def random_like(e, r=16):
    out = []
    for s in e:
        kk = min(r, s.shape[-1]); g = torch.Generator(device=s.device)
        g.manual_seed(int(s.float().abs().sum().item() * 1e3) % (2 ** 31))
        A = torch.randn(*s.shape[:-1], kk, device=s.device, dtype=torch.float32, generator=g)
        B = torch.randn(*s.shape[:-2], kk, s.shape[-1], device=s.device, dtype=torch.float32, generator=g)
        M = A @ B
        out.append((M * (s.float().norm() / (M.norm() + 1e-9))).to(s.dtype))
    return out


def score(state, test_letters, k):
    ok = 0
    for x in test_letters:
        c = fwd(ids(PRIMER, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s%s" % (x, SEP)), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == caesar(x, k)
    return ok / len(test_letters)


clean, neutral, notrans, rnd, wrong, full = [], [], [], [], [], []
for s in range(SEEDS):
    rng = random.Random(7000 + s)
    kw = rng.choice([d for d in range(1, 13) if d != K])     # a different shift
    L = rng.sample(list(AL), NTRAIN + NTEST)
    train, test = L[:NTRAIN], L[NTRAIN:]
    trainY = rng.sample(list(AL), NTRAIN)
    try:
        eX = engram(demos(train, K))             # the k-rule engram
        eXw = engram(demos(train, kw))           # a wrong-shift engram
        eY = engram(demos(trainY, K))            # recipient holds the same rule on other letters
        c_clean = score(eX, test, K)             # engram alone, neutral recipient (ceiling)
        c_rnd = score(random_like(eX), test, K)  # random engram alone
        c_wrong = score(eXw, test, K)            # wrong-shift engram alone
        c_full = score(add(eY, eX), test, K)     # superposed onto a same-rule recipient
        c_nt = score(eY, test, K)                # recipient alone (also knows the rule!) -> high
    except Exception as e:
        sys.stderr.write("skip %d: %r\n" % (s, e)); continue
    clean.append(c_clean); rnd.append(c_rnd); wrong.append(c_wrong)
    full.append(c_full); notrans.append(c_nt)


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("model:", mid, " rule=Caesar k=%d  reps=%d ntrain=%d ntest=%d seeds=%d (chance~0.038)"
      % (K, REPS, NTRAIN, NTEST, len(clean)))
for nm, d in [("engram alone (rule ceiling)", clean), ("random engram alone", rnd),
              ("wrong-shift engram alone", wrong), ("recipient alone (knows rule)", notrans),
              ("superposed transfer", full)]:
    m, lo, hi = boot(d)
    print("  %-30s %.3f  [%.3f, %.3f]" % (nm, m, lo, hi))
for nm, A, B in [("clean - random", clean, rnd), ("clean - wrong-shift", clean, wrong)]:
    df = np.array(A) - np.array(B); rng = np.random.default_rng(0)
    bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
    print("  %-22s d=%.3f [%.3f, %.3f] p=%.4f" %
          (nm, df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))
