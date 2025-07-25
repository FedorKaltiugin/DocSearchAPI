import os
import time
import math
import gdown
import nltk
import torch
import py7zr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
nltk.download("punkt")

app = FastAPI()

load_dotenv()

model_rag_id = os.getenv("model_rag")
model_trained_id = os.getenv("model_trained")

os.makedirs("models", exist_ok=True)
rag_archive = "models/flan-t5-lora-rag.7z"
trained_archive = "models/flan-t5-lora-trained.7z"

gdown.download(f"https://drive.google.com/uc?id={model_rag_id}", rag_archive, quiet=False)
gdown.download(f"https://drive.google.com/uc?id={model_trained_id}", trained_archive, quiet=False)

with py7zr.SevenZipFile(rag_archive, mode='r') as z:
    z.extractall(path="models")
os.remove(rag_archive)

with py7zr.SevenZipFile(trained_archive, mode='r') as z:
    z.extractall(path="models")
os.remove(trained_archive)

dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'question-answer', split='test')

bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
token_embeddings = []
batch_size = 1000
for i in range(0, len(dataset), batch_size):
    batch_questions = dataset['question'][i:i + batch_size]
    batch_embeddings = bert_model.encode(batch_questions, batch_size=64)
    token_embeddings.append(batch_embeddings)
embeddings = np.vstack(token_embeddings)

base_model_name = 'models/flan-t5-lora-trained'
model = T5ForConditionalGeneration.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, 'models/flan-t5-lora-trained')
tokenizer = T5Tokenizer.from_pretrained('models/flan-t5-lora-trained')
model.eval()


def bm25_search(query, dataset, top_n=5):
    tokenized_query = word_tokenize(query.lower())
    passages = [word_tokenize(q.lower()) for q in dataset['question']]
    bm25 = BM25Okapi(passages)
    scores = bm25.get_scores(tokenized_query)
    top_ids = np.argsort(scores)[::-1][:top_n]
    return [{'question': dataset['question'][int(i)], 'answer': dataset['answer'][int(i)], 'score': float(scores[i])}
            for i in top_ids]


def bert_search(query, dataset, embeddings, top_n=5):
    query_vec = bert_model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_ids = np.argsort(scores)[::-1][:top_n]
    return [{'question': dataset['question'][int(i)], 'answer': dataset['answer'][int(i)], 'score': float(scores[i])}
            for i in top_ids]


def normalize_scores(score_dict):
    values = list(score_dict.values())
    min_s, max_s = min(values), max(values)
    return {k: (v - min_s) / (max_s - min_s) if max_s > min_s else 0.0 for k, v in score_dict.items()}


def hybrid_search(query, dataset, embeddings, top_n=5):
    bm25 = bm25_search(query, dataset, top_n=100)
    bert = bert_search(query, dataset, embeddings, top_n=100)
    bm25_scores = {r['question']: r['score'] for r in bm25}
    bert_scores = {r['question']: r['score'] for r in bert}
    bm25_norm = normalize_scores(bm25_scores)
    bert_norm = normalize_scores(bert_scores)
    all_questions = set(bm25_norm) | set(bert_norm)

    merged = []
    for q in all_questions:
        score = 0.6 * bm25_norm.get(q, 0) + 0.4 * bert_norm.get(q, 0)
        ans = next((r['answer'] for r in bm25 + bert if r['question'] == q), '')
        merged.append({'question': q, 'answer': ans, 'score': round(score, 4)})

    return sorted(merged, key=lambda x: -x['score'])[:top_n]


def compute_perplexity(results, model, tokenizer):
    total_loss = 0
    total_tokens = 0
    for sample in results:
        input_text = f'Answer the question: {sample["question"]}'
        answer_text = sample['answer']

        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        targets = tokenizer(text_target=[answer_text], return_tensors='pt', truncation=True, padding='max_length',
                            max_length=32)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = targets['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    return round(math.exp(avg_loss), 4)


class SearchRequest(BaseModel):
    query: str
    method: Literal["bm25", "bert", "hybrid"]
    top_n: int = 5


@app.post('/search')
def search(req: SearchRequest):
    start = time.time()

    if req.method == 'bm25':
        results = bm25_search(req.query, dataset, req.top_n)
    elif req.method == 'bert':
        results = bert_search(req.query, dataset, embeddings, req.top_n)
    else:
        results = hybrid_search(req.query, dataset, embeddings, req.top_n)

    perplexity = compute_perplexity(results, model, tokenizer)
    elapsed = round(time.time() - start, 4)

    return {
        'query': req.query,
        'method': req.method,
        'top_n': req.top_n,
        'time_seconds': elapsed,
        'perplexity': perplexity,
        'results': results
    }


if __name__ == '__main__':
    import uvicorn

    # uvicorn.run(app, host='0.0.0.0', port=8000)
    uvicorn.run(app, host='127.0.0.1', port=8000)
