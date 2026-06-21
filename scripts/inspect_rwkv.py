#!/usr/bin/env python3
"""Load RWKV-7 via fla and inspect its cache / recurrent state structure, to see
whether the engrammics LMBackend (which reads/writes per-layer recurrent_state)
can be adapted to it. Also a one-shot greedy sanity decode."""
import os, sys
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_ID"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
print("loading", mid)
try:
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt,
                                                 trust_remote_code=True).to(dev).eval()
    print("loaded via trust_remote_code")
except Exception as e:
    print("remote_code load failed:", repr(e)[:300])
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval()
    print("loaded via fla native")

print("model class:", model.__class__.__name__)
ids = tok("A=1;B=2;C=3;", return_tensors="pt").input_ids.to(dev)
print("token ids:", ids.tolist())
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
cache = out.past_key_values
print("cache type:", type(cache))
print("len(cache):", end=" ")
try:
    print(len(cache))
except Exception as e:
    print("fail", e)
# inspect layer 0
layer0 = None
try:
    layer0 = cache[0]
except Exception as e:
    print("cache[0] failed:", e)
print("layer0 type:", type(layer0))
if isinstance(layer0, dict):
    for k, v in layer0.items():
        if isinstance(v, torch.Tensor):
            print(f"  key={k!r} shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, (tuple, list)):
            print(f"  key={k!r} {type(v).__name__} len={len(v)}; " +
                  ", ".join(f"[{i}]{tuple(x.shape)}" for i, x in enumerate(v)
                            if isinstance(x, torch.Tensor)))
        else:
            print(f"  key={k!r} = {type(v)}")
else:
    for a in ["recurrent_state", "conv_state", "attn_state", "ffn_state", "state"]:
        v = getattr(layer0, a, "MISSING")
        if isinstance(v, torch.Tensor):
            print(f"  attr={a} shape={tuple(v.shape)}")
        else:
            print(f"  attr={a} = {type(v) if v!='MISSING' else 'MISSING'}")
nxt = int(out.logits[0, -1].argmax())
print("greedy next after 'A=1;B=2;C=3;':", repr(tok.decode([nxt])))
