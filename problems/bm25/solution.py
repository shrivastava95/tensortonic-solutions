import numpy as np
from collections import Counter, defaultdict
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if len(docs) == 0:
        return []
    
    token_counts_per_doc = [Counter(doc) for doc in docs]
    token_counts = Counter()
    for c in token_counts_per_doc:
        token_counts.update(c)

    token_to_id_map = {}
    for k, v in token_counts.items():
        token_to_id_map[k] = len(token_to_id_map)
    T = len(token_to_id_map)

    tf_td = np.zeros([T, N])
    for did, (tcounter) in enumerate(token_counts_per_doc):
        for token, tdcount in tcounter.items():
            tid = token_to_id_map[token]
            tf_td[tid, did] += tdcount
    
    DL = np.sum(tf_td, axis=0).reshape(1, -1)
    avgdl = np.mean(DL).item()
    avgdl = avgdl if avgdl > 0 else 1

    df_t = np.sum(tf_td > 0, axis=1).reshape(-1, 1)

    idf_t = np.log( (N - df_t + 0.5) / (df_t + 0.5) + 1 )

    # for a query that never appears inside the corpus, set idf to 0
    query_tokens = [qt for qt in query_tokens if qt in token_to_id_map]
    
    query_tokens = list(np.unique(query_tokens))
    
    # remove all indices apart from queries for calculation
    query_token_indices = [token_to_id_map[token] for token in query_tokens]
    tf_td = tf_td[query_token_indices, :]
    idf_t = idf_t[query_token_indices, :]

    score_dq = idf_t * ( ( tf_td*(k1+1) ) / ( tf_td+k1*(1-b+b*(DL/avgdl)) ) )
    score_dq_queried = np.sum(score_dq, axis=0)
    return score_dq_queried
    

