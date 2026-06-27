#!/usr/bin/env python3
"""Discover a PURE DeltaNet checkpoint on fla-hub and print its identifier.
No id is hard-coded: the hub is queried at runtime. Override possible via the
MODEL_ID environment variable."""
import sys

try:
    from huggingface_hub import list_models
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"[pick_model] huggingface_hub missing: {e}\n")
    print("")
    sys.exit(0)

cands = []
try:
    for m in list_models(author="fla-hub"):
        n = m.id.lower()
        # pure DeltaNet: drop the 'gated' variants and 'attn' hybrids
        if "delta" in n and "gated" not in n and "attn" not in n:
            cands.append(m.id)
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"[pick_model] list_models failed: {e}\n")


def score(i):
    s = i.lower()
    pref = 0 if ("1.3b" in s or "-1b" in s or "1b" in s) else 1  # aim for ~1B
    return (pref, len(s))


cands.sort(key=score)
print(cands[0] if cands else "")
