# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""
- GitHub에는 이 wrapper만 공개
"""

import argparse
try:
    from private_train import train_and_maybe_merge, infer_merged
except ImportError:
    raise RuntimeError("⚠️ private_train 모듈이 로컬에는 존재해야 합니다. (GitHub에는 없음)")

# ---- CLI Wrappers ----
def run_train(args):
    train_and_maybe_merge(
        excel_path=args.excel,
        kakao_jsonl=args.kakao,
        stage1_dir=args.stage1_dir,
        final_dir=args.final_dir,
        do_merge=args.merge
    )

def run_infer(args):
    infer_merged(
        model_dir=args.final_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

def main():
    parser = argparse.ArgumentParser(description="JayChat Trainer CLI")
    sub = parser.add_subparsers(dest="cmd")

    # 학습
    p_train = sub.add_parser("train")
    p_train.add_argument("--excel", required=True)
    p_train.add_argument("--kakao", required=True)
    p_train.add_argument("--stage1_dir", default="stage1_adapter")
    p_train.add_argument("--final_dir", default="final_adapter")
    p_train.add_argument("--merge", action="store_true")
    p_train.set_defaults(func=run_train)

    # 추론
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--final_dir", required=True)
    p_infer.add_argument("--prompt", required=True)
    p_infer.add_argument("--max_new_tokens", type=int, default=256)
    p_infer.add_argument("--temperature", type=float, default=0.7)
    p_infer.add_argument("--top_p", type=float, default=0.95)
    p_infer.set_defaults(func=run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
