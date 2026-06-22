import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR   = Path(__file__).resolve().parent.parent
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
rules      = open(BASE_DIR / "data/finance_rules.txt", encoding="utf-8").read().splitlines()
embeddings = np.load(BASE_DIR / "data/rule_embeddings.npy")

def retrieve_rules(query: str, top_k: int = 3) -> list[dict]:
    query_emb = embedder.encode([query])
    scores    = cosine_similarity(query_emb, embeddings)[0]
    top_idx   = scores.argsort()[::-1][:top_k]
    return [
        {"rule": rules[i], "score": round(float(scores[i]), 3)}
        for i in top_idx
    ]

if __name__ == "__main__":
    test_queries = [
        "I spend too much on food delivery",
        "How much should I save every month",
        "My rent is very high",
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        for r in retrieve_rules(q):
            print(f"  [{r['score']}] {r['rule']}")