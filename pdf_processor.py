import hashlib
import io
from pathlib import Path
from urllib.parse import unquote

import fitz  # PyMuPDF

# ==========================================
# SETTINGS
# ==========================================

PDF_FOLDER = Path("data/pdfs")
OUTPUT_FILE = Path("data/pdf_text.txt")

# PDFs whose name contains any of these are skipped (club events, contests...).
# Remove an item from this list if you want that PDF in the chatbot.
EXCLUDE = [
    "NDLI", "ndli", "readabookday", "World_Poetry", "worldthinking",
    "QUIZ_CONTEST", "Report_Bookmark", "Report_Eco", "Report_Finding",
    "Report_on_session", "Report_story", "Report_Tricolor", "report_behindpen",
    "Library_Shelfie", "The_book_quest", "Uniquness", "worlsday", "Competition_",
]

OCR_MIN_CHARS = 30          # a page with fewer characters than this is treated as scanned
OCR_DPI = 200
TESSERACT_PATH = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"       # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==========================================
# OCR SETUP (optional)
# ==========================================

OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image

    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    OCR_AVAILABLE = True
except ImportError:
    print("NOTE: pytesseract/Pillow not installed -> scanned PDFs will stay empty.")


def extract_page_text(page):
    """Normal text first; OCR only if the page has (almost) no text layer."""
    global OCR_AVAILABLE

    text = page.get_text().strip()
    if len(text) >= OCR_MIN_CHARS or not OCR_AVAILABLE:
        return text

    try:
        pixmap = page.get_pixmap(dpi=OCR_DPI)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image).strip()
    except Exception as error:
        print("   OCR unavailable:", error)
        OCR_AVAILABLE = False      # stop trying, don't spam errors
        return text


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    if not PDF_FOLDER.exists():
        print("PDF folder does not exist!")
        return

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    print("PDF files found:", len(pdf_files))

    seen_hashes = set()
    documents = []
    empty_pdfs = []
    skipped = []

    for number, pdf_file in enumerate(pdf_files, start=1):
        name = unquote(pdf_file.name)
        print(f"\n[{number}/{len(pdf_files)}] {name}")

        if any(word in pdf_file.name for word in EXCLUDE):
            print("   skipped (event/club report)")
            skipped.append(name)
            continue

        digest = file_hash(pdf_file)
        if digest in seen_hashes:
            print("   skipped (duplicate of another PDF)")
            skipped.append(name)
            continue
        seen_hashes.add(digest)

        try:
            document = fitz.open(pdf_file)
            pages = [extract_page_text(page) for page in document]
            document.close()

            pages = [p for p in pages if p.strip()]
            if not pages:
                print("   WARNING: no text found")
                empty_pdfs.append(name)
                continue

            documents.append(
                "\n\n" + "=" * 80 + "\n"
                + f"SOURCE PDF: {name}\n"
                + "=" * 80 + "\n"
                + "\n".join(pages)
            )
            print(f"   OK ({len(pages)} pages with text)")

        except Exception as error:
            print("   ERROR:", error)

    OUTPUT_FILE.write_text("\n".join(documents), encoding="utf-8")

    print("\n" + "=" * 60)
    print("PDF PROCESSING COMPLETED")
    print("=" * 60)
    print("PDFs saved to text :", len(documents))
    print("PDFs skipped       :", len(skipped))
    print("PDFs with NO text  :", len(empty_pdfs))
    for name in empty_pdfs:
        print("   -", name)
    if empty_pdfs:
        print("\n(Install Tesseract + pytesseract and run again to read these scanned PDFs.)")


if __name__ == "__main__":
    main()
