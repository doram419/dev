from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner = pipeline("ner", model=model, tokenizer=tokenizer)
example = "서울역으로 안내해줘."
ner_results = ner(example)
print(ner_results)