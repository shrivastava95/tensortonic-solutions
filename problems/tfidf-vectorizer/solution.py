import numpy as np
from collections import Counter, defaultdict
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    doc_splitted = [doc.lower().split() for doc in documents]
    if len(doc_splitted) == 0:
        return np.array([]), []
    elif sum([len(item) for item in doc_splitted]) == 0:
        return np.array([]), []
    uniques = []
    term_id_map = {}
    counts_term_doc_sparse = defaultdict(lambda: defaultdict(int))
    for did, doc in enumerate(doc_splitted):
        if len(doc) == 0:
            continue # skip for empty documents, not needed to build the sparse counts
        for term in doc:
            if term not in term_id_map:
                term_id_map[term] = len(term_id_map)
            tid = term_id_map[term]
            counts_term_doc_sparse[tid][did] += 1
    counts_term_doc = np.zeros([len(term_id_map), len(doc_splitted)])
    for tid, temp in counts_term_doc_sparse.items():
        for did, tdcount in temp.items():
            counts_term_doc[tid, did] += tdcount

    counts_doc_term = np.transpose(counts_term_doc, (1, 0)) # doc, term
    counts_doc = np.sum(counts_term_doc, axis=0).reshape(-1, 1) # doc, 1
    counts_doc = np.where(counts_doc > 0, counts_doc, 1)

    tf_td = counts_doc_term / counts_doc

    df_t = np.sum(counts_doc_term > 0, axis=0).reshape(1, -1) # 1, term
    idf_t = np.log(len(doc_splitted)) - np.log(df_t) # 1, term

    # print(df_t.shape, idf_t.shape)
    tfidf_matrix = tf_td * idf_t    # doc, term
    vocabulary = list(term_id_map.keys())

    # sorting
    sorted_indices = np.argsort(vocabulary)
    sorted_vocabulary = [vocabulary[i] for i in sorted_indices]
    sorted_tfidf_matrix = tfidf_matrix[:, sorted_indices]
    return sorted_tfidf_matrix, sorted_vocabulary

