import torch
import math
from app.utils.logger import logger


def generate_llm_answer(query, results, model, tokenizer):
    logger.info("Generating answer with LLM...")
    context = "\n".join([f"Q: {r['question']} A: {r['answer']}" for r in results])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    logger.debug(f"Constructed prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    logger.info("Prompt tokenized and moved to model device")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    logger.info("Answer generated")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug(f"Decoded answer: {answer}")
    return answer


def compute_perplexity(query, context, answer, model, tokenizer):
    logger.info("Computing perplexity for the generated answer")
    prompt = f"{context}\n\nQuestion: {query}\nAnswer: {answer}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    logger.debug(f"Prompt for perplexity: {prompt}")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    perplexity = math.exp(outputs.loss.item())
    logger.info(f"Perplexity computed: {perplexity:.4f}")
    return perplexity
