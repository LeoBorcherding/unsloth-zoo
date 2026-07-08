import torch
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=488, load_in_4bit=False, fast_inference=True,
    gpu_memory_utilization=0.30, dtype=torch.float16, max_lora_rank=16)
m = FastLanguageModel.get_peft_model(m, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing=False, random_state=0)

# make the LoRA non-trivial (B is zero-init) so the merge has a detectable effect
for n,p in m.named_parameters():
    if "lora_B" in n: p.data.normal_(0, 0.02)

hf = m.base_model.model.model            # HF training model (frozen base + LoRA)
vm = m.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model.model  # vLLM Llama

def base_w(proj): return getattr(proj,"base_layer",proj).weight
def hf_mods(layer):  # (name, hf_module) for the 7 targets
    a=layer.self_attn; mlp=layer.mlp
    return {"q":a.q_proj,"k":a.k_proj,"v":a.v_proj,"o":a.o_proj,
            "gate":mlp.gate_proj,"up":mlp.up_proj,"down":mlp.down_proj}
def delta(mod):
    A=mod.lora_A.default.weight; B=mod.lora_B.default.weight; s=mod.scaling["default"]
    return (B.float() @ A.float()).mul_(s)

# record HF base norms to prove no drift
pre = {n: base_w(p).norm().item() for n,p in
       [(f"{i}.{k}", hf_mods(hf.layers[i])[k]) for i in (0, len(hf.layers)//2) for k in ("q","gate","down")]}

# ---- MERGE: overwrite vLLM base slices = HF_base + delta (HF base read-only) ----
QKV=[("q",slice(0,3072)),("k",slice(3072,4096)),("v",slice(4096,5120))]
GU =[("gate",slice(0,8192)),("up",slice(8192,16384))]
for i,vl in enumerate(vm.layers):
    hm = hf_mods(hf.layers[i])
    qkv=base_w(vl.self_attn.qkv_proj); gu=base_w(vl.mlp.gate_up_proj)
    for nm,sl in QKV: qkv[sl].copy_((base_w(hm[nm]).float()+delta(hm[nm])).to(qkv.dtype))
    for nm,sl in GU:  gu[sl].copy_((base_w(hm[nm]).float()+delta(hm[nm])).to(gu.dtype))
    ow=base_w(vl.self_attn.o_proj); ow.copy_((base_w(hm["o"]).float()+delta(hm["o"])).to(ow.dtype))
    dw=base_w(vl.mlp.down_proj);    dw.copy_((base_w(hm["down"]).float()+delta(hm["down"])).to(dw.dtype))
torch.cuda.synchronize()

# ---- CORRECTNESS: vLLM slice == HF_base + delta ? ----
print("=== merge correctness (max abs err, vLLM base slice vs HF_base+delta) ===")
for i in (0, len(hf.layers)//2, len(hf.layers)-1):
    hm=hf_mods(hf.layers[i]); qkv=base_w(vm.layers[i].self_attn.qkv_proj)
    for nm,sl in QKV:
        exp=(base_w(hm[nm]).float()+delta(hm[nm]))
        err=(qkv[sl].float()-exp).abs().max().item()
        print(f"  L{i} {nm}: {err:.2e}")
    dw=base_w(vm.layers[i].mlp.down_proj)
    err=(dw.float()-(base_w(hm['down']).float()+delta(hm['down']))).abs().max().item()
    print(f"  L{i} down: {err:.2e}")

print("=== no-drift: HF base norm before/after ===")
post={n: base_w(p).norm().item() for n,p in
      [(f"{i}.{k}", hf_mods(hf.layers[i])[k]) for i in (0, len(hf.layers)//2) for k in ("q","gate","down")]}
for k in pre: print(f"  {k}: pre {pre[k]:.6f} post {post[k]:.6f} same={pre[k]==post[k]}")
print("TEST_MERGE_DONE")
