# Clean LoRA -> vLLM merge (no drift) — design

## Goal
Speed up GRPO vLLM generation by feeding vLLM **merged** weights (base+LoRA, no
per-token adapter math) — the way TRL gets its faster decode — but **without** TRL's
numerical drift.

## Why TRL drifts (what NOT to do)
TRL (`trl/trainer/grpo_trainer.py::_move_model_to_vllm`) does, every step:
```
self.model.merge_adapter()     # W += scale * B@A   (in place, on the training base)
llm_model.load_weights(...)    # copy merged W into vLLM
self.model.unmerge_adapter()   # W -= scale * B@A   (in place)
```
`W += d` then `W -= d` in fp16/bf16 does not perfectly cancel, so the frozen base `W`
drifts a little every step and never recovers -> silent corruption over a long RL run.

## The clean version
Never write to the training base. Compute the merged weight **into vLLM's own buffer**
from the untouched training base each step:
```
W_vllm = W_base_train + scale * (B @ A)      # read W_base_train, never modify it
```
- `W_base_train` is the frozen base in the training model — read only, so it can't drift.
- `W_vllm` is vLLM's own weight tensor — a full **overwrite** each step (copy_), not `+=`,
  so no accumulation. vLLM then decodes with plain merged weights (no adapter) = faster.
- No persistent 2nd model beyond vLLM's existing weight buffer.

## Where it plugs in (this repo)
- Replace the per-step call to `load_lora_directly(model)` / the `LoRARequest` path in the
  GRPO sync (`rl_replacements.py` hook + `vllm_utils.py::load_lora*`) with a new
  `merge_lora_into_vllm(model)` and run vLLM with `enable_lora=False`.
- Reuse the existing vLLM base-weight mapping in `vllm_utils.py::get_vllm_state_dict` /
  `get_state_dict` — it already resolves each projection's vLLM tensor and handles the
  **fused** `qkv_proj` (q=0,k=1,v=2 slices) and `gate_up_proj`, plus fp8/bnb. That mapping
  is exactly what tells us WHERE to write each merged projection.

## Per-module math
For each target module (q,k,v,o,gate,up,down):
```
delta = (lora_B @ lora_A) * scaling            # scaling = lora_alpha / r
vllm_base_slice.copy_(train_base_weight + delta.to(vllm_base_slice.dtype))
```
- q/k/v write into their slice of the fused `qkv_proj`; gate/up into `gate_up_proj`.
- keep everything in the vLLM weight dtype; do the add in fp32 then cast for less error.
- 4-bit base (bnb) needs dequant->merge->requant (out of scope for the fp16 GRPO path;
  gate on `load_in_4bit`).

## Risk / testing
A wrong fused offset or a missing transpose **silently** produces garbage generations (no
crash). MUST be validated on GPU:
1. correctness: after merge, a vLLM `generate` on merged weights == HF forward with the
   adapter applied (logits match within fp tolerance) for a few prompts.
2. no-drift: hash/`norm` the training base before and after N steps — must be identical.
3. speed: GRPO steady sec/step vs the current adapter path (expect vLLM-gen 1.03 -> ~0.6s).

## Status
Prototype `merge_lora_into_vllm` scaffold added to `vllm_utils.py` (clearly marked
UNTESTED). It reuses the existing mapping but the fused-offset writes need live GPU
verification before trusting. Not wired into the GRPO step yet.
