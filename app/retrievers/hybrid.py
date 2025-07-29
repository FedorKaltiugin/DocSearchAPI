from app.retrievers.bm25 import bm25_search
from app.retrievers.bert import bert_search
from app.utils.logger import logger


def normalize_scores(score_dict):
    values = list(score_dict.values())
    min_s, max_s = min(values), max(values)
    return {k: (v - min_s) / (max_s - min_s) if max_s > min_s else 0.0 for k, v in score_dict.items()}


def hybrid_search(query, dataset, embeddings, bert_model, top_n=5):
    logger.info(f"Starting hybrid search for query: '{query}'")

    try:
        bm25 = bm25_search(query, dataset, top_n=100)
        bert = bert_search(query, dataset, embeddings, bert_model, top_n=100)

        logger.debug(f"BM25 returned {len(bm25)} results")
        logger.debug(f"BERT returned {len(bert)} results")

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

        merged_sorted = sorted(merged, key=lambda x: -x['score'])[:top_n]
        logger.info(f"Hybrid search completed. Top {top_n} results selected.")
        for i, item in enumerate(merged_sorted):
            logger.debug(f"{i+1}. Score: {item['score']} | Q: {item['question']}")

        return merged_sorted

    except Exception as e:
        logger.exception(f"Error during hybrid search: {e}")
        raise
