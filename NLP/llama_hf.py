import torch
import transformers

# GPU 사용 가능 여부를 확인
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA memory cache cleared.")

# 사용할 모델의 ID 설정 (Hugging Face 모델 허브에서 다운로드 가능한 모델 이름)
# model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Meta에서 제공하는 Llama 3.1 모델 (8B 파라미터, Instruct 튜닝)
model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"

# 텍스트 생성 파이프라인 초기화
pipeline = transformers.pipeline(
    "text-generation",  # 파이프라인의 작업 타입: 텍스트 생성
    model=model_id,  # 사용할 모델의 ID
    model_kwargs={"torch_dtype": torch.bfloat16},  # 모델을 bfloat16(효율적인 메모리 사용)으로 로드
    device_map="auto",  # 모델을 자동으로 적절한 장치(CPU/GPU)에 배치
)

# 대화형 입력 메시지 정의
messages = [
    {"role": "system",
     "content": "모든 대답은 한국어(Korean)으로 대답해주세요."},
    {"role": "user", "content": "대한민국의 수도에 대해 설명해주세요"},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 텍스트 생성 수행
outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# 생성된 텍스트의 마지막 문자를 출력
print(outputs[0]["generated_text"][-1]["content"])
