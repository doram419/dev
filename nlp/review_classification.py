from transformers import pipeline

text = "인생은 가까이서 보면 비극이지만, 멀리서 보면 희극이다"

# pipeline("테스크 명", "모델명")
classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")
result = classifier(text)
print(result)
