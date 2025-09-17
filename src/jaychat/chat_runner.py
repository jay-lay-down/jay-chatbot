# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "Upstage/SOLAR-10.7B-Instruct-v1.0"

def _is_adapter(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "adapter_config.json"))

def _has_full(model_dir: str) -> bool:
    names = ["pytorch_model.bin", "model.safetensors", "consolidated.safetensors"]
    return any(os.path.exists(os.path.join(model_dir, n)) for n in names) \
           and os.path.exists(os.path.join(model_dir, "config.json"))

def _load(model_dir: str, tokenizer_dir: str | None = None):
    tk_dir = tokenizer_dir or (model_dir if os.path.exists(os.path.join(model_dir,"tokenizer.json")) or os.path.exists(os.path.join(model_dir,"tokenizer.model")) else BASE_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(tk_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if _is_adapter(model_dir) and not _has_full(model_dir):
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_dir)
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
        model.eval()
        return model, tok

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tok

def run_chat(args):
    model, tok = _load(args.model_dir, args.tokenizer_dir)
    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=args.temperature, top_p=args.top_p,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))
