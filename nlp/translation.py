from transformers import pipeline

pipe = pipeline("translation", model="circulus/kobart-trans-ko-en-v2")
result = pipe("너 왜 내 연락 안 받아?")
print(result)