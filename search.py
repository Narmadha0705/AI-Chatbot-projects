import json
import pickle
import re
from pathlib import Path
 
import faiss
import numpy as np
from fastembed import TextEmbedding
 
VECTOR_FOLDER = Path("vector_db")
 
WEB_BOOST = 0.06
MODE_BOOST = 0.10
 
REGULATION_WORDS = (
    "regulation", "attendance", "arrear", "credit", "passing", "pass mark", "grade", "cgpa",
    "re-joining", "rejoin", "ordinance", "syllabus", "examination", "exam", "ph.d", "phd",
    "research", "supplementary", "revaluation", "malpractice",
)
 
# (pattern in the question, text that must appear in the document label with spaces/dots removed)
MODE_PATTERNS = [
    (r"part[\s-]?time", "parttime"),
    (r"working[\s-]?professional", "workingprofessional"),
    (r"\bm\.e\b|\bm\.?\s?tech\b|\bpg\b|post[\s-]?graduate", "mtech"),
    (r"ph\.?\s?d\b|doctoral", "phd"),
]
 
_index = None
_chunks = None
_model = None
_query_prefix = ""
 
 
def _load():
    """Load everything once, on the first search."""
    global _index, _chunks, _model, _query_prefix
 
    if _index is not None:
        return
 
    meta = json.loads((VECTOR_FOLDER / "meta.json").read_text(encoding="utf-8"))
    _query_prefix = meta["query_prefix"]
    _model = TextEmbedding(model_name=meta["model"])
    _index = faiss.read_index(str(VECTOR_FOLDER / "university.index"))
 
    with open(VECTOR_FOLDER / "chunks.pkl", "rb") as file:
        _chunks = pickle.load(file)
 
 
def _label_key(text):
    """'[Source: Regulations 2021 M E M Tech Part Time]\\n...' -> 'regulations2021memmtechparttime'"""
    label = text.split("]", 1)[0]
    return re.sub(r"[\s.\-_]+", "", label.lower())
 
 
def search_knowledge(question, top_k=5):
    """-> [{"text": chunk_text, "score": similarity}, ...] best first"""
    _load()
 
    query = np.array(list(_model.embed([_query_prefix + question])), dtype="float32")
    query /= np.linalg.norm(query, axis=1, keepdims=True)
 
    # look at many candidates, then re-order with the boosts
    scores, ids = _index.search(query, max(top_k * 6, 40))
 
    q = question.lower()
    boost_web = not any(word in q for word in REGULATION_WORDS)
    wanted_modes = [key for pattern, key in MODE_PATTERNS if re.search(pattern, q)]
 
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        text = _chunks[idx]
        score = float(score)
 
        if boost_web and text.startswith("[Source: http"):
            score += WEB_BOOST
 
        label = _label_key(text)
        for key in wanted_modes:
            if key in label:
                score += MODE_BOOST
 
        results.append({"text": text, "score": score})
 
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
 
 
if __name__ == "__main__":
    # quick test:  python search.py
    for question in ("What is the admission procedure for UG?",
                     "What is the attendance requirement for M.E. part time?"):
        print(question)
        for r in search_knowledge(question, top_k=4):
            print(f"  {r['score']:.3f}  {r['text'][:80]!r}")
 
