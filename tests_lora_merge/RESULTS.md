# Live results (MI300X, Llama-3.2-3B, fast_inference, 2026-07-08)

## test_share.py — is vLLM's base weight shared with the HF training base?
    hf_q data_ptr    : 122682730872832
    vllm_qkv q-slice : 122682730872832
    same storage?    : True
    after writing 123 to vLLM q-slice[0,0], HF q[0,0] = 123.0 (was -0.0327)
    => SHARED

## test_merge.py — merge layout correctness + drift
    merge correctness (vLLM slice vs HF_base+delta): max err ~5e-3  (= fp16 storage rounding; layout correct)
    no-drift: HF base norms CHANGED after merging into vLLM  (because base is shared)

Conclusion: base is stored once and shared between training + vLLM, so merging into
vLLM corrupts the training base. Merged-decode would require a separate base copy
(+1 full model in VRAM). See ../DESIGN_lora_merge_no_drift.md.
