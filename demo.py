
from transformers import pipeline

pl = pipeline("text-generation", model="medalpaca-7b", tokenizer="medalpaca-7b")
question = "What are the symptoms of diabetes?"
context = "Diabetes is a metabolic disease that causes high blood sugar. The symptoms include increased thirst, frequent urination, and unexplained weight loss."
answer = pl(f"Context: {context}\n\nQuestion: {question}\n\nAnswer: ")
print(answer)
