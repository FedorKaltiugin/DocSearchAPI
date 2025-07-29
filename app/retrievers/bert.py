from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.utils.logger import logger


def bert_search(query, dataset, embeddings, bert_model, top_n=5):
    logger.info(f"Starting BERT search for query: '{query}'")

    try:
        query_vec = bert_model.encode([query])
        scores = cosine_similarity(query_vec, embeddings)[0]
        top_ids = np.argsort(scores)[::-1][:top_n]

        results = [
            {
                'question': dataset['question'][int(i)],
                'answer': dataset['answer'][int(i)],
                'score': float(scores[i])
            }
            for i in top_ids
        ]

        logger.info("BERT search completed successfully")
        return results

    except Exception as e:
        logger.exception(f"Error during BERT search: {e}")
        raise
