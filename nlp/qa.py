from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
model = AutoModelForQuestionAnswering.from_pretrained("yjgwak/klue-bert-base-finetuned-squard-kor-v1")
tokenizer = AutoTokenizer.from_pretrained("yjgwak/klue-bert-base-finetuned-squard-kor-v1")

question = "총 얼마가 있을까?"
context = "왼손에는 100만원 오른손엔 1000만원이 있다"

input = tokenizer(question, context, return_tensors="pt")
outputs = model(**input)
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# print output text
print(tokenizer.decode(input["input_ids"][0][torch.argmax(start_scores):torch.argmax(end_scores)+1]))