"""Debug: can Unsloth load Qwen3.5-4B in bf16 with LoRA and give last-token hidden states?"""
import time

import torch
from unsloth import FastLanguageModel

t = time.time()
model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, load_in_4bit=False,
                                               max_seq_length=512)
print("loaded", type(model).__name__, f"{time.time() - t:.0f}s", torch.cuda.memory_allocated() / 2**30)
names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
print("linear module names", names)
targets = [n for n in names if n != "lm_head"]
model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=16, lora_dropout=0.05, target_modules=targets,
                                         use_rslora=True, use_gradient_checkpointing="unsloth", random_state=0)
model.print_trainable_parameters()
inner = model.base_model.model.model
print("inner", type(inner).__name__)
tk = getattr(tok, "tokenizer", tok)
ids = tk(["Hello there, is this sarcastic?"], return_tensors="pt").to(0)
out = inner(**ids, use_cache=False)
h = out.last_hidden_state[:, -1]
print("hidden", h.shape, h.dtype, float(h.float().norm()))
h.float().sum().backward()
print("backward ok; lora grads:", sum(p.grad is not None for n, p in model.named_parameters() if "lora_" in n))
