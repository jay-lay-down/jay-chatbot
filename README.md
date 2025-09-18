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
✅ 미세조정: 자주 사용하는 단어는 화이트리스트/자주 사용하지 않는 어색한 단어는 블랙리스트 구현 후, 몇 가지 일상적 질문에 대한 예시 답변 제시

# 테스트 
This is a fine-tuned SOLAR model for conversational AI tasks.

✅ 모델 테스트(Python에서 실행)\

```bash
pip install transformers accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "jay1121/solar-chatbot-final"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("안녕?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

## 참고
GPU 필요 (Colab T4/A100 권장)

# 실제 채팅 예시
![Chat Example 1](./assets/image.png)
![Chat Example 2](./assets/image (1).png)

