import torch
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=488, load_in_4bit=False, fast_inference=True,
    gpu_memory_utilization=0.30, dtype=torch.float16, max_lora_rank=16)
m = FastLanguageModel.get_peft_model(m, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing=False, random_state=0)
hf = m.base_model.model.model
vm = m.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model.model
def bw(p): return getattr(p,"base_layer",p).weight
hf_q = bw(hf.layers[0].self_attn.q_proj)          # (3072,3072)
v_qkv = bw(vm.layers[0].self_attn.qkv_proj)        # (5120,3072), q=[0:3072]
print("hf_q data_ptr   :", hf_q.data_ptr())
print("vllm_qkv q-slice:", v_qkv[0:3072].data_ptr())
print("same storage?   :", hf_q.data_ptr()==v_qkv[0:3072].data_ptr())
orig = hf_q[0,0].item()
v_qkv[0,0] = 123.0; torch.cuda.synchronize()
print("after writing 123 to vLLM q-slice[0,0], HF q[0,0] =", hf_q[0,0].item(), "(was", orig,")")
print("SHARED" if hf_q[0,0].item()==123.0 else "SEPARATE")
print("TEST_SHARE_DONE")
