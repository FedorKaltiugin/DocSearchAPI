import nltk
import time
import numpy as np
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal
from sentence_transformers import SentenceTransformer
from app.utils.logger import logger

from app.data.loader import download_and_extract_models, load_dataset_from_hub
from app.models.load_model import load_llm_model
from app.llm.ask import generate_llm_answer, compute_perplexity
from app.retrievers.bm25 import bm25_search
from app.retrievers.bert import bert_search
from app.retrievers.hybrid import hybrid_search

nltk.download("punkt")

logger.info("Starting the server and initializing components")

app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    method: Literal["bm25", "bert", "hybrid"] = "hybrid"
    top_n: int = 5


download_and_extract_models()
dataset = load_dataset_from_hub()

bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = np.vstack([
    bert_model.encode(dataset['question'][i:i + 1000])
    for i in range(0, len(dataset), 1000)
])

model, tokenizer = load_llm_model('models/flan-t5-lora-trained')


@app.post('/search')
def search(request: SearchRequest):
    start = time.time()

    if request.method == "bm25":
        results = bm25_search(request.query, dataset, request.top_n)
    elif request.method == "bert":
        results = bert_search(request.query, dataset, embeddings, bert_model, request.top_n)
    else:
        results = hybrid_search(request.query, dataset, embeddings, bert_model, request.top_n)

    answer = generate_llm_answer(request.query, results, model, tokenizer)
    context = "\n".join([f"Q: {r['question']} A: {r['answer']}" for r in results])
    perplexity = compute_perplexity(request.query, context, answer, model, tokenizer)
    end = time.time()

    return {
        "answer": answer,
        "time": round(end - start, 2),
        "perplexity": round(perplexity, 2)
    }


static_path = os.path.join(os.path.dirname(__file__), "static")

app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(static_path, "index.html"))
