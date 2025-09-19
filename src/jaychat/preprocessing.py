# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import os, re, json
from tqdm.auto import tqdm

# 여러 카카오톡 내보내기 포맷 대응
CHAT_PATTERNS = [
    re.compile(r'^\w+\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2},\s+(.*?)\s+:\s+(.*)'),
    re.compile(r'^\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*(오전|오후)\s*\d{1,2}:\d{2},\s+(.*?)\s*:\s*(.*)'),
    re.compile(r'^\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*(오전|오후)\s*\d{1,2}:\d{2},\s+(.*?)\s*:\s*(.*)'),
]

def match_line(line: str):
    for pat in CHAT_PATTERNS:
        m = pat.match(line)
        if m:
            if len(m.groups()) == 2:
                speaker, message = m.groups()
            elif len(m.groups()) == 3:
                _, speaker, message = m.groups()
            else:
                continue
            return speaker, message
    return None

def parse_and_clean_chats(chats_dir, my_name, names_to_remove=None):
    print(f"--- 데이터 정제 시작 (내 이름: '{my_name}', 폴더: {chats_dir}) ---")
    if not os.path.isdir(chats_dir):
        print(f"❌ 오류: '{chats_dir}' 폴더를 찾을 수 없습니다.")
        return []

    all_dialogues = []
    chat_files = [f for f in os.listdir(chats_dir) if f.lower().endswith('.txt')]
    if not chat_files:
        print(f"❌ 오류: '{chats_dir}' 폴더에 분석할 .txt 파일이 없습니다.")
        return []

    for filename in tqdm(sorted(chat_files), desc="대화 파일 처리 중"):
        filepath = os.path.join(chats_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print("읽기 실패:", filename, e)
            continue
        
        dialogue_turns = []; last_speaker = None; buf = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if ("대화가 시작" in line) or ("저장한 날짜" in line):
                continue

            m = match_line(line)
            if m:
                speaker, message = m
                if speaker != last_speaker and last_speaker is not None:
                    if buf:
                        dialogue_turns.append({"speaker": last_speaker, "message": "\n".join(buf).strip()})
                    buf = []
                buf.append(message)
                last_speaker = speaker
            else:
                if buf:
                    buf.append(line)

        if last_speaker and buf:
            dialogue_turns.append({"speaker": last_speaker, "message": "\n".join(buf).strip()})
        all_dialogues.extend(dialogue_turns)

    training_pairs = []
    keywords_to_filter = ["링크", "이미지", "삭제된 메시지입니다", "동영상", "선물을 보냈습니다", "사진"]

    for i in range(1, len(all_dialogues)):
        cur = all_dialogues[i]; prev = all_dialogues[i-1]
        if cur['speaker'] == my_name and prev['speaker'] != my_name:
            if prev['message'] and cur['message']:
                opp = prev['message']; mine = cur['message']
                if any(k in opp for k in keywords_to_filter) or any(k in mine for k in keywords_to_filter):
                    continue
                opp  = re.sub(r'\b' + re.escape(my_name) + r'\b', 'ME', opp)
                mine = mine.replace(my_name, 'ME')
                if names_to_remove:
                    for n in names_to_remove:
                        if not n: continue
                        opp  = re.sub(r'\b' + re.escape(n) + r'\b', 'Friend', opp)
                        mine = mine.replace(n, 'Friend')
                final_text = f"### User:\n{opp}\n\n### Assistant:\n{mine}"
                training_pairs.append({"text": final_text})

    print(f"✅ 총 {len(training_pairs)}개의 데이터 쌍을 추출했습니다.")
    return training_pairs

def run_preprocess(args):
    names_to_censor = [s.strip() for s in (args.names or "").split(",")] if args.names else []
    data = parse_and_clean_chats(args.chats_dir, args.my_name, names_to_censor)
    if not data:
        raise SystemExit("❌ 추출된 데이터가 없습니다. txt 포맷/이름 설정을 확인하세요.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for pair in data:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"✅ 저장: {args.out}")
