# jay-chatbot
나의 말투와 응답 패턴을 모방하는 맞춤형 챗봇

# 사용 데이터
✅ 데이터 수집 기간: 2023년 6월 - 2025년 8월\
✅ 수집 범위: 카카오톡 백업 파일\
✅ 데이터 변환: 텍스트/csv 파일을 .json 확장자로 변환\
✅ 사용 언어: Python 3.12\
✅ 사용 LLM: Solar 10.7B\
✔ 허깅페이스 링크: SOLAR-10.7B-v1.0

## 🛠️ 학습 과정

### ✅ 환경 준비
- **모델:** SOLAR 10.7B  
- **학습 방법:** 4bit 양자화 + QLoRA (경량화 및 효율적 미세조정)  

---

### ✅ 1단계 학습 (패턴 학습)
- 몇 가지 Q&A 예시 문답을 학습시켜  
  → 사용자의 **기본 말투·응답 패턴** 학습  

---

### ✅ 2단계 학습 (카카오톡 전체 데이터 적용)
- 카카오톡 백업 파일 전체를 학습  
- 실제 **자주 사용하는 단어/문체/응답 스타일** 구현  

---

### ✅ 미세조정 (Whitelist/Blacklist 적용)
- 자주 쓰는 단어는 **Whitelist**, 어색한 단어는 **Blacklist** 처리  
- 일상 질문에 대한 **예시 답변 추가 (few-shot)**  

# 테스트 
Python (Colab) 환경에서 바로 테스트 가능

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

## Content License

- **Code** is licensed under **Apache-2.0**.
- **Documentation, examples, and other non-code content** are licensed under **CC BY-NC-ND 4.0**.

You may change the content license to **CC BY 4.0** if you want to allow commercial use and derivatives with attribution.

## No-impersonation & Branding

This project is not affiliated with or endorsed by any person or organization.
You may not use my name, likeness, voice, or distinctive writing style to imply endorsement or to build systems that impersonate me without explicit permission.


