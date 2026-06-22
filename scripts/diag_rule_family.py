#!/usr/bin/env python3
"""Does the engram carry a FAMILY of induced rules, or just one Caesar shift?
This generalizes diag_caesar_transfer.py to several structurally distinct rules,
on a model that can induce them in context (RWKV-7-2.9B), each scored on HELD-OUT
inputs never demonstrated. For each rule we write its engram from a train set,
inject it (recurrent state only, neutral out-of-domain primer) into a neutral
recipient, and compare to a random engram and a WRONG-RULE engram (same structure,
different transformation). 'engram alone' above both controls = the specific
induced rule transferred, not a memorized table and not the engram's shape.

RULE env (default runs the whole suite if RULE unset):
  caesar:k   letters, x -> (x+k) mod 26      (translation; k!=1 is not a prior)
  atbash     letters, x -> 25-x              (REFLECTION: structurally not a shift)
  digit:k    digits,  x -> (x+k) mod 10      (different domain/alphabet)
Env: REPS(3) NTRAIN(10) NTEST(8) SEEDS(50)."""
import os, sys, random
import numpy as np
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
REPS = int(os.environ.get("REPS", 3))
NTRAIN, NTEST = int(os.environ.get("NTRAIN", 10)), int(os.environ.get("NTEST", 8))
SEEDS = int(os.environ.get("SEEDS", 50))
LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIG = "0123456789"
SEP = ">"
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


class Rule:
    """A rule = an alphabet, a transform f, a neutral primer (out-of-domain), and a
    'wrong' sibling rule of the same structure for the specificity control."""
    def __init__(self, name, alpha, f, primer, wrong):
        self.name, self.alpha, self.f, self.primer, self.wrong = name, alpha, f, primer, wrong


def make_rule(spec):
    if spec.startswith("caesar:"):
        k = int(spec.split(":")[1])
        kw = 7 if k != 7 else 4
        return Rule(spec, LET, lambda x, k=k: LET[(LET.index(x) + k) % 26],
                    "1>1;2>2;3>3;", lambda x, kw=kw: LET[(LET.index(x) + kw) % 26])
    if spec == "atbash":
        return Rule(spec, LET, lambda x: LET[25 - LET.index(x)],
                    "1>1;2>2;3>3;", lambda x: LET[(LET.index(x) + 5) % 26])  # wrong = a shift
    if spec.startswith("digit:"):
        k = int(spec.split(":")[1])
        kw = 7 if k != 7 else 4
        return Rule(spec, DIG, lambda x, k=k: DIG[(DIG.index(x) + k) % 10],
                    "a>a;b>b;c>c;", lambda x, kw=kw: DIG[(DIG.index(x) + kw) % 10])
    raise ValueError(spec)


def demos(letters, f): return "".join("%s%s%s;" % (x, SEP, f(x)) for x in letters) * REPS
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


def score(state, test, f, primer):
    ok = 0
    for x in test:
        c = fwd(ids(primer, bos=True)).past_key_values
        setrec(c, state)
        o = fwd(ids("%s%s" % (x, SEP)), cache=c)
        ok += tok.decode([int(o.logits[0, -1].argmax())]).strip() == f(x)
    return ok / len(test)


def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def run_rule(spec):
    R = make_rule(spec)
    pool = R.alpha
    clean, rnd, wrong = [], [], []
    ntr = min(NTRAIN, len(pool) - NTEST) if len(pool) < NTRAIN + NTEST else NTRAIN
    nte = min(NTEST, len(pool) - ntr)
    for s in range(SEEDS):
        rng = random.Random(9000 + s)
        L = rng.sample(list(pool), ntr + nte)
        train, test = L[:ntr], L[ntr:]
        try:
            eX = engram(demos(train, R.f))
            clean.append(score(eX, test, R.f, R.primer))
            rnd.append(score(random_like(eX), test, R.f, R.primer))
            wrong.append(score(engram(demos(train, R.wrong)), test, R.f, R.primer))
        except Exception as e:
            sys.stderr.write("skip %s seed %d: %r\n" % (spec, s, e)); continue
    chance = 1.0 / len(pool)
    print("\n=== rule=%s  (alphabet=%d, chance~%.3f, %d seeds) ==="
          % (spec, len(pool), chance, len(clean)))
    for nm, d in [("engram alone (induced+transferred)", clean),
                  ("random engram", rnd), ("wrong-rule engram", wrong)]:
        m, lo, hi = boot(d)
        print("  %-34s %.3f  [%.3f, %.3f]" % (nm, m, lo, hi))
    for nm, A, B in [("clean - random", clean, rnd), ("clean - wrong-rule", clean, wrong)]:
        df = np.array(A) - np.array(B); rng = np.random.default_rng(0)
        bd = df[rng.integers(0, len(df), size=(10000, len(df)))].mean(1)
        print("  %-20s d=%.3f [%.3f, %.3f] p=%.4f" %
              (nm, df.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5), float((bd <= 0).mean())))


SUITE = os.environ.get("RULE")
specs = [SUITE] if SUITE else ["caesar:3", "caesar:5", "atbash", "digit:3"]
print("model:", mid, " reps=%d seeds=%d  RULE FAMILY: %s" % (REPS, SEEDS, ", ".join(specs)))
for spec in specs:
    run_rule(spec)
print("\nRULE_FAMILY_DONE")
