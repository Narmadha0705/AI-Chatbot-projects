"""
Search module.
 
Fix: the interactive test loop is now inside `if __name__ == "__main__":`,
so `from search import search_knowledge` no longer starts a loop by accident.
 
Test the search on its own with:  python search.py
"""
 
import json
import pickle
from pathlib import Path
 
import faiss
from sentence_transformers import SentenceTransformer
 
VECTOR_FOLDER = Path("vector_db")
 
index = faiss.read_index(str(VECTOR_FOLDER / "university.index"))
 
with open(VECTOR_FOLDER / "chunks.pkl", "rb") as file:
    chunks = pickle.load(file)
 
# the model name is saved by build_knowledge_base.py so both always match
meta = json.loads((VECTOR_FOLDER / "meta.json").read_text(encoding="utf-8"))
QUERY_PREFIX = meta["query_prefix"]
model = SentenceTransformer(meta["model"])
 
 
def search_knowledge(question, top_k=5):
    """Return the top_k most similar chunks (score = cosine similarity, higher is better)."""
    embedding = model.encode(
        [QUERY_PREFIX + question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
 
    scores, indices = index.search(embedding, top_k)
 
    results = []
    for rank, (score, chunk_number) in enumerate(zip(scores[0], indices[0]), start=1):
        if chunk_number == -1:
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "text": chunks[chunk_number],
        })
    return results
 
 
if __name__ == "__main__":
    print("Knowledge base ready! Vectors:", index.ntotal)
 
    while True:
        question = input("\nAsk (type exit to stop): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
 
        for result in search_knowledge(question, top_k=5):
            print(f"\nRESULT {result['rank']}   score = {result['score']:.3f}")
            print(result["text"][:1200])
            print("-" * 70)
 