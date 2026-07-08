import os, torch
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=488, load_in_4bit=False, fast_inference=True,
    gpu_memory_utilization=0.30, dtype=torch.float16, max_lora_rank=16)
m = FastLanguageModel.get_peft_model(m, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing=False, random_state=0)

print("=== unsloth_zoo helpers present? ===")
import unsloth_zoo.vllm_utils as vu
for fn in ["get_vllm_state_dict","load_lora_directly","load_lora","get_state_dict","assert_same_state_dict"]:
    print(" ", fn, hasattr(vu, fn))
print("=== model attrs (loras/vllm) ===")
for a in ["vllm_engine","model_loras_A","model_loras_B","vllm_loras_A","vllm_loras_B"]:
    print(" ", a, hasattr(m, a))

print("=== reach vLLM inner model ===")
eng = getattr(m,"vllm_engine",None)
paths=[]
try:
    vm = eng.llm_engine.model_executor.driver_worker.model_runner.model
    print("  V0 path OK:", type(vm).__name__)
except Exception as e:
    print("  V0 path fail:", str(e)[:120])
    try:
        vm = eng.llm_engine.model_executor.driver_worker.worker.model_runner.model
        print("  alt path OK:", type(vm).__name__)
    except Exception as e2:
        print("  alt fail:", str(e2)[:120]); vm=None
if vm is not None:
    lyr = vm.model.layers[0]
    qkv = lyr.self_attn.qkv_proj; gu = lyr.mlp.gate_up_proj
    print("  qkv_proj:", type(qkv).__name__, "weight", tuple(qkv.weight.shape), qkv.weight.dtype)
    print("  gate_up_proj:", type(gu).__name__, "weight", tuple(gu.weight.shape))
    for attr in ["output_sizes","output_size","q_size","kv_size","num_heads","input_size"]:
        print("   qkv.",attr,"=",getattr(qkv,attr,"n/a"))

print("=== HF training model LoRA (layer 0 q_proj) ===")
sd = {k:v.shape for k,v in m.state_dict().items() if "layers.0.self_attn.q_proj" in k}
for k,v in sd.items(): print("  ",k,tuple(v))
# scaling
for n,mod in m.named_modules():
    if n.endswith("layers.0.self_attn.q_proj") and hasattr(mod,"scaling"):
        print("  q_proj.scaling:", getattr(mod,"scaling",None)); break
print("EXPLORE_DONE")
