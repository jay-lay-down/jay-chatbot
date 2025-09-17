from __future__ import annotations
import argparse
from .preprocessing import run_preprocess
from .train_qlora import run_train, run_infer
from .chat_runner import run_chat

def main():
    ap = argparse.ArgumentParser(
        prog="jaychat",
        description="KakaoTalk preprocessing + Two-stage QLoRA + inference"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # preprocess
    sp = sub.add_parser("preprocess", help="카카오톡 txt 폴더 → jsonl({\"text\": ...})")
    sp.add_argument("--chats-dir", required=True, help="카톡 txt들이 들어있는 폴더")
    sp.add_argument("--my-name", required=True, help="내 카톡 표시 이름(정확히)")
    sp.add_argument("--names", default="", help="추가 마스킹 이름(콤마로 구분)")
    sp.add_argument("--out", required=True, help="생성할 jsonl 경로")
    sp.set_defaults(func=lambda a: run_preprocess(a))

    # train
    st = sub.add_parser("train", help="2-Stage QLoRA (엑셀 Friend/Me → 카톡 jsonl)")
    st.add_argument("--excel", required=True, help="Friend/Me 엑셀 경로(xlsx)")
    st.add_argument("--kakao", required=True, help="preprocess로 만든 jsonl 경로")
    st.add_argument("--stage1-dir", required=True, help="1단계 어댑터 저장 폴더")
    st.add_argument("--final-dir", required=True, help="최종(어댑터/병합) 저장 폴더")
    st.add_argument("--merge", action="store_true", help="학습 후 LoRA 병합 저장")
    st.set_defaults(func=lambda a: run_train(a))

    # infer (merged model)
    si = sub.add_parser("infer", help="병합된 단일 모델로 생성")
    si.add_argument("--final-dir", required=True, help="병합 모델 폴더")
    si.add_argument("--prompt", required=True, help="생성 프롬프트(전체)")
    si.add_argument("--max-new-tokens", type=int, default=256)
    si.add_argument("--temperature", type=float, default=0.7)
    si.add_argument("--top-p", type=float, default=0.95)
    si.set_defaults(func=lambda a: run_infer(a))

    # chat (adapter/full 자동 판별)
    sc = sub.add_parser("chat", help="모델 로드 후 한 번 생성(또는 간단 채팅)")
    sc.add_argument("--model-dir", required=True, help="어댑터 폴더 또는 병합 모델 폴더")
    sc.add_argument("--tokenizer-dir", default=None, help="토크나이저 위치(선택)")
    sc.add_argument("--prompt", default="### User:\n안녕?\n\n### Assistant:\n")
    sc.add_argument("--max-new-tokens", type=int, default=128)
    sc.add_argument("--temperature", type=float, default=0.7)
    sc.add_argument("--top-p", type=float, default=0.95)
    sc.set_defaults(func=lambda a: run_chat(a))

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
