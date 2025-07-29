from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
from app.utils.logger import logger


def bm25_search(query, dataset, top_n=5):
    logger.info(f"Starting BM25 search for query: '{query}'")

    try:
        tokenized_query = word_tokenize(query.lower())
        passages = [word_tokenize(q.lower()) for q in dataset['question']]
        bm25 = BM25Okapi(passages)
        scores = bm25.get_scores(tokenized_query)
        top_ids = np.argsort(scores)[::-1][:top_n]

        logger.info(f"Top {top_n} results selected: {top_ids.tolist()}")

        results = [
            {
                'question': dataset['question'][int(i)],
                'answer': dataset['answer'][int(i)],
                'score': float(scores[i])
            }
            for i in top_ids
        ]

        logger.info("BM25 search completed successfully")
        return results

    except Exception as e:
        logger.exception(f"Error during BM25 search: {e}")
        raise
