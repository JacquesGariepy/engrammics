#!/usr/bin/env python3
"""Decouvre un checkpoint DeltaNet PUR sur fla-hub et imprime son identifiant.
Aucun id n'est code en dur : on interroge le hub a l'execution. Override possible
via la variable d'environnement MODEL_ID."""
import sys

try:
    from huggingface_hub import list_models
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"[pick_model] huggingface_hub manquant: {e}\n")
    print("")
    sys.exit(0)

cands = []
try:
    for m in list_models(author="fla-hub"):
        n = m.id.lower()
        # DeltaNet pur : on ecarte les variantes 'gated' et hybrides 'attn'
        if "delta" in n and "gated" not in n and "attn" not in n:
            cands.append(m.id)
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"[pick_model] echec list_models: {e}\n")


def score(i):
    s = i.lower()
    pref = 0 if ("1.3b" in s or "-1b" in s or "1b" in s) else 1  # vise ~1B
    return (pref, len(s))


cands.sort(key=score)
print(cands[0] if cands else "")
