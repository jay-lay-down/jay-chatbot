# jay-chatbot
나의 말투를 모방하는 챗봇

# 사용 데이터
✅ 데이터 수집 기간: 2023년 6월 - 2025년 8월\
✅ 수집 범위: 카카오톡 백업 파일\
✅ 데이터 변환: 텍스트/csv 파일을 .json 확장자로 변환\
✅ 사용 언어: Python 3.12\
✅ 사용 LLM: Solar 10.7B\
✔ 허깅페이스 링크: SOLAR-10.7B-v1.0

# 학습 과정
✅ 학습방법: 4bit 양자화 + QLoRA\
✅ 1단계 학습: 몇 가지 Q & A 문답을 학습시켜, 사용자의 말투/응답 패턴을 학습시킴\
✅ 2딘계 학습: 카카오톡 백업 파일을 한 번 더 학습시킨 후 실제 사용하는 단어/말투 구현\
✅ 미세조정: 자주 사용하는 단어는 화이트리스트/자주 사용하지 않는 어색한 단어는 블랙리스트 구현 후, 몇 가지 일상적 질문에 대한 예시 답변 제시\

# 테스트 

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -e .

## 전처리
jaychat preprocess --chats_dir ./kakao_txt --my_name "사용자 이름 넣기" --out kakaotalk.jsonl --confirm

## 학습(QLoRA)
jaychat train --excel my_dataset.xlsx --kakao kakaotalk.jsonl --merge

## 추론(merged)
jaychat infer --model-dir checkpoints/my-solar-chatbot-final --prompt "오늘 뭐 해?"

## 테스트 시작
jaychat chat --model-dir checkpoints/my-solar-chatbot-final
```

## 참고
GPU 필요 (Colab T4/A100 권장)

# 실제 채팅 예시
![Chat Example 1](./assets/image.png)
![Chat Example 2](./assets/image (1).png)

