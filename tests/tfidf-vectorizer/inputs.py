import numpy as np

tests = {
    1: {
        "solver": "tfidf_vectorizer", 
        "args": [], 
        "kwargs": dict(documents=["the cat sat", "the dog ran"]), 
        "output": 'Output: matrix shape (2, 5), vocab=["cat", "dog", "ran", "sat", "the"]',
    }
}
