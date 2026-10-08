import json
import pickle
import re
from pathlib import Path

WEBSITE_FILE = Path("data/website.txt")
PDF_FILE = Path("data/pdf_text.txt")
VECTOR_FOLDER = Path("vector_db")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 80
BATCH_SIZE = 64


# ==========================================
# READING THE TWO DATA FILES
# ==========================================

def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_website_pages(text):
    """-> [(url, content), ...]"""
    pattern = r"SOURCE:\s*(https?://\S+)[ \t]*\n(.*?)(?=\n\s*SOURCE:\s*https?://|\Z)"
    return [(url.strip(), content) for url, content in re.findall(pattern, text, flags=re.DOTALL)]


def pdf_label(name):
    """'Admission_procedure_01_04754f5529 - Copy (2).pdf' -> 'Admission procedure 01'"""
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(\s*-\s*Copy(\s*\(\d+\))?)+\s*$", "", name, flags=re.IGNORECASE)  # " - Copy (2)"
    name = re.sub(r"_[0-9a-f]{10}$", "", name)        # hash added by the old downloader
    return name.replace("_", " ").strip()


def read_pdf_documents(text):
    """-> [(label, content), ...]"""
    pattern = r"={80}\nSOURCE PDF: (.*?)\n={80}\n(.*?)(?=\n={80}\nSOURCE PDF: |\Z)"
    return [(pdf_label(name), content) for name, content in re.findall(pattern, text, flags=re.DOTALL)]


# ==========================================
# MAIN
# ==========================================

def main():
    # heavy imports here so the helper functions above can be tested without them
    import faiss
    import numpy as np
    from fastembed import TextEmbedding                      # ONNX based, no torch needed
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    VECTOR_FOLDER.mkdir(exist_ok=True)

    documents = []   # (label, text)

    if WEBSITE_FILE.exists():
        pages = read_website_pages(WEBSITE_FILE.read_text(encoding="utf-8"))
        print("Website pages:", len(pages))
        documents += pages
    else:
        print("website.txt not found - run crawler.py first!")

    if PDF_FILE.exists():
        pdfs = read_pdf_documents(PDF_FILE.read_text(encoding="utf-8"))
        print("PDF documents:", len(pdfs))
        documents += pdfs
    else:
        print("pdf_text.txt not found - run pdf_processor.py first!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    seen = set()

    for label, content in documents:
        content = clean_text(content)
        for piece in splitter.split_text(content):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS or piece in seen:
                continue
            seen.add(piece)
            chunks.append(f"[Source: {label}]\n{piece}")

    print("Total chunks:", len(chunks))

    if not chunks:
        print("No chunks were created - check your data files. Nothing saved.")
        return

    print("\nLoading embedding model (first time downloads ~130 MB)...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Creating embeddings...")
    vectors = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors.extend(model.embed(batch, batch_size=BATCH_SIZE))
        print(f"  {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}", end="\r")
    print()

    embeddings = np.array(vectors, dtype="float32")

    # normalise so inner product == cosine similarity
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # inner product on normalised vectors == cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(VECTOR_FOLDER / "university.index"))

    with open(VECTOR_FOLDER / "chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

    (VECTOR_FOLDER / "meta.json").write_text(
        json.dumps({"model": MODEL_NAME, "query_prefix": QUERY_PREFIX}),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE CREATED")
    print("=" * 60)
    print("Chunks :", len(chunks))
    print("Vectors:", index.ntotal)


if __name__ == "__main__":
    main()