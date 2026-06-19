import numpy as np
from sentence_transformers import SentenceTransformer

def embed_rules():
    with open("data/finance_rules.txt") as f:
        rules = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Embedding {len(rules)} financial rules...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(rules, show_progress_bar=True)

    np.save("data/rule_embeddings.npy", embeddings)

    print(f"✅ Saved {len(rules)} rule embeddings")
    print(f"   Embeddings shape: {embeddings.shape}")
    return rules, embeddings

if __name__ == "__main__":
    embed_rules()