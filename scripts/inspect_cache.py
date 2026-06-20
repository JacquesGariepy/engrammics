#!/usr/bin/env python3
"""Introspect the flash-linear-attention cache returned by a DeltaNet forward
pass so we can correctly adapt LMBackend._iter/_get/_set. Prints the cache type,
how layer states are stored, and the keys/shapes of one layer state."""
import os, sys
import torch
import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_ID", "")
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
print(f"model={mid} device={dev} dtype={dt}")

tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval()

ids = tok("A=3;B=7;", return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
cache = out.past_key_values
print("cache type:", type(cache))
print("has .states attr:", hasattr(cache, "states"))
print("dir (public):", [a for a in dir(cache) if not a.startswith("__")][:40])

# try to find the per-layer container
st = getattr(cache, "states", None)
print("states is", type(st), "len" , (len(st) if st is not None else None))
try:
    print("len(cache):", len(cache))
except Exception as e:
    print("len(cache) failed:", e)

# inspect first layer state
layer0 = None
if isinstance(st, (list, tuple)) and len(st):
    layer0 = st[0]
else:
    try:
        layer0 = cache[0]
    except Exception as e:
        print("cache[0] failed:", e)

print("layer0 type:", type(layer0))
if isinstance(layer0, dict):
    for k, v in layer0.items():
        if isinstance(v, torch.Tensor):
            print(f"  key={k!r}  tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  key={k!r}  {type(v)} = {v}")
else:
    for a in ["recurrent_state", "conv_state", "attn_state"]:
        v = getattr(layer0, a, "MISSING")
        if isinstance(v, torch.Tensor):
            print(f"  attr={a}  tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  attr={a} = {v}")

# greedy next-token sanity
nxt = int(out.logits[:, -1, :].argmax(-1))
print("greedy next token after 'A=3;B=7;':", repr(tok.decode([nxt])))
