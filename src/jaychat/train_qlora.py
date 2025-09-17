# -*- coding: utf-8 -*-
import os, random, inspect
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, set_seed
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

BASE_MODEL_PATH = "Upstage/SOLAR-10.7B-Instruct-v1.0"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 128, 0.10
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
EPOCHS, BATCH, ACCUM = 2, 8, 4
LR1, LR2 = 5e-5, 2e-5
MAX_LEN = 512
WARMUP = 0.03

def _bnb_cfg():
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)

def _load_base_4bit():
    bnb = _bnb_cfg()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = prepare_model_for_kbit_training(model)
    return model, tok

def _lora_cfg():
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")

def _train_args(out, lr):
    return TrainingArguments(
        output_dir=out, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
        gradient_checkpointing=True, optim="paged_adamw_8bit",
        learning_rate=lr, bf16=True, fp16=False, weight_decay=0.0, max_grad_norm=0.3,
        warmup_ratio=WARMUP, lr_scheduler_type="cosine",
        logging_steps=50, save_strategy="epoch", save_safetensors=True, save_total_limit=2,
        report_to="none", ddp_find_unused_parameters=False
    )

def _make_trainer(model, tok, ds, peft_cfg, args):
    params = inspect.signature(SFTTrainer.__init__).parameters
    kw = dict(model=model, train_dataset=ds, peft_config=peft_cfg, args=args)
    if "processing_class" in params: kw["processing_class"] = tok
    elif "tokenizer" in params:     kw["tokenizer"] = tok
    if   "max_seq_length" in params: kw["max_seq_length"] = MAX_LEN
    elif "max_length" in params:     kw["max_length"] = MAX_LEN
    if "remove_unused_columns" in params: kw["remove_unused_columns"] = False
    if "packing" in params: kw["packing"] = False
    if "dataset_text_field" in params:
        kw["dataset_text_field"] = "text"
        return SFTTrainer(**kw)
    else:
        def fmt(ex):
            t = ex.get("text","")
            if isinstance(t, list): t = t[0] if t else ""
            return str(t)
        kw["formatting_func"] = fmt
        return SFTTrainer(**kw)

def _excel_to_jsonl(excel_path, out_path="/tmp/excel_stage1.jsonl"):
    df = pd.read_excel(excel_path)
    if not {"Friend","Me"}.issubset(df.columns):
        raise ValueError("엑셀은 Friend/Me 두 열이 필요합니다.")
    def clean(x):
        if pd.isna(x): return None
        s = str(x).strip()
        bad = ["이모티콘","사진을 보냈습니다","삭제된 메시지"]
        return None if (not s or any(b in s for b in bad)) else s
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            u, a = clean(r["Friend"]), clean(r["Me"])
            if u and a:
                f.write((f'{{"text": "### User:\\n{u}\\n\\n### Assistant:\\n{a}"}}\n'))
                n += 1
    print(f"STAGE1 jsonl: {out_path} (pairs={n})")
    return out_path

def train_and_maybe_merge(excel_path, kakao_jsonl, stage1_dir, final_dir, do_merge):
    set_seed(42); random.seed(42); np.random.seed(42)

    # Stage 1
    s1_jsonl = _excel_to_jsonl(excel_path)
    ds1 = load_dataset("json", data_files=s1_jsonl, split="train")
    model, tok = _load_base_4bit()
    peft = _lora_cfg()
    tr1 = _make_trainer(model, tok, ds1, peft, _train_args("/tmp/results_s1", LR1))
    tr1.train()
    tr1.model.save_pretrained(stage1_dir); tok.save_pretrained(stage1_dir)
    del tr1, model; torch.cuda.empty_cache()

    # Stage 2
    ds2 = load_dataset("json", data_files=kakao_jsonl, split="train")
    model, _ = _load_base_4bit()
    model = PeftModel.from_pretrained(model, stage1_dir)
    tok = AutoTokenizer.from_pretrained(stage1_dir)
    tr2 = _make_trainer(model, tok, ds2, _lora_cfg(), _train_args("/tmp/results_final", LR2))
    tr2.train()
    tr2.model.save_pretrained(final_dir); tok.save_pretrained(final_dir)

    if do_merge:
        print("Merging LoRA into base weights ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(final_dir, safe_serialization=True)
        tok.save_pretrained(final_dir)

    del tr2, model; torch.cuda.empty_cache()
    print(f"✅ saved: stage1={stage1_dir}, final={final_dir}")

def infer_merged(model_dir, prompt, max_new_tokens=256, temperature=0.7, top_p=0.95):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature, top_p=top_p,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))

# ---- CLI wrappers ----
def run_train(args):
    train_and_maybe_merge(args.excel, args.kakao, args.stage1_dir, args.final_dir, args.merge)

def run_infer(args):
    infer_merged(args.final_dir, args.prompt, args.max_new_tokens, args.temperature, args.top_p)
