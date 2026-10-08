import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, unquote
 
import requests
from bs4 import BeautifulSoup
 
# ==========================================
# SETTINGS
# ==========================================
 
START_URL = "https://avit.ac.in"
MIN_PAGE_CHARS = 100       # pages with less text than this are dropped (and listed)
MAX_PAGES = 300            # raise if the site has more useful pages
DELAY_SECONDS = 0.5        # be polite to the server
DOWNLOAD_PDFS = False      # you already have the PDFs; set True to fetch new ones
 
# URLs we never want
SKIP_IN_URL = ("/blogs/", "/download", "/search", "/gallery", "/wp-admin", "/wp-json", "/feed",
               "/tag/", "/author/", "/login", "/cart")
SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
                   ".zip", ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx",
                   ".ppt", ".pptx", ".css", ".js")
 
# HTML tags that never contain real content
JUNK_TAGS = ["script", "style", "noscript", "nav", "footer", "header",
             "aside", "form", "iframe", "svg", "button"]
 
# If an element's class/id contains one of these, it is probably layout, not content
JUNK_HINTS = ("menu", "navbar", "nav-", "footer", "sidebar", "breadcrumb",
              "marquee", "ticker", "popup", "modal", "cookie", "social",
              "topbar", "top-bar", "announcement")
 
# A line that appears on this share of all pages is boilerplate (menu, banner...)
BOILERPLATE_SHARE = 0.4
 
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
 
DATA_FOLDER = Path("data")
PDF_FOLDER = DATA_FOLDER / "pdfs"
 
 
# ==========================================
# URL HELPERS
# ==========================================
 
def normalize_url(url):
    """Drop #fragment, ?query and trailing slash so the same page isn't visited twice."""
    url, _ = urldefrag(url)
    url = url.split("?")[0]
    parts = urlparse(url)
    if parts.netloc.startswith("www."):
        url = parts._replace(netloc=parts.netloc[4:]).geturl()
    return url.rstrip("/")
 
 
def same_domain(url):
    base = urlparse(START_URL).netloc.replace("www.", "")
    return urlparse(url).netloc.replace("www.", "") == base
 
 
def wanted(url):
    low = url.lower()
    if not same_domain(url):
        return False
    if any(word in low for word in SKIP_IN_URL):
        return False
    if low.endswith(SKIP_EXTENSIONS):
        return False
    return True
 
 
# ==========================================
# PAGE CLEANING
# ==========================================
 
def tables_to_text(soup):
    """Replace every <table> with one text line per row, keeping column names."""
    for table in reversed(soup.find_all("table")):
        headers, lines = [], []
 
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
 
            header_row = row.find("th") is not None and row.find("td") is None
            if header_row and not headers:
                headers = cells
                continue
 
            if headers and len(headers) == len(cells):
                lines.append(" | ".join(f"{h}: {c}" for h, c in zip(headers, cells)))
            else:
                lines.append(" | ".join(cells))
 
        replacement = soup.new_tag("div")
        replacement.string = "\n".join(lines)
        table.replace_with(replacement)
 
 
def remove_layout_elements(soup):
    """Remove menu/sidebar/banner style elements (only if they are a small part of the page)."""
    total = len(soup.get_text(" ", strip=True)) or 1
 
    for element in soup.find_all(True):
        try:
            if element.decomposed:
                continue
            if element.name in ("html", "body", "main", "article"):
                continue
 
            label = " ".join(element.get("class", [])) + " " + (element.get("id") or "")
            label = label.lower()
 
            if any(hint in label for hint in JUNK_HINTS):
                # safety check: never delete something that is most of the page
                if len(element.get_text(" ", strip=True)) < 0.5 * total:
                    element.decompose()
        except Exception:
            continue
 
 
def page_to_lines(content):
    """Raw HTML bytes -> (title, list of clean text lines)."""
    # from_encoding fixes the â¹ / â problem
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
 
    title = soup.title.get_text(strip=True) if soup.title else ""
 
    # the <meta name="description"> often holds the only real summary of thin pages (e.g. labs)
    description = ""
    meta = (soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"}))
    if meta and meta.get("content"):
        description = re.sub(r"\s+", " ", meta["content"]).strip()
 
    for tag in soup(JUNK_TAGS):
        tag.decompose()
 
    remove_layout_elements(soup)
    tables_to_text(soup)
 
    root = soup.body or soup
    text = root.get_text("\n", strip=True)
 
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
 
    if description and description not in lines:
        lines.insert(0, description)
 
    return title, lines
 
 
def remove_repeated_lines(pages):
    """
    pages = [(url, title, lines), ...]
    Delete lines that appear on many pages (menus, banners, contact strips).
    """
    count = len(pages)
    if count < 5:
        return pages
 
    frequency = Counter()
    for _, _, lines in pages:
        frequency.update(set(lines))
 
    limit = max(3, int(count * BOILERPLATE_SHARE))
 
    cleaned = []
    for url, title, lines in pages:
        kept = [line for line in lines if frequency[line] < limit]
        cleaned.append((url, title, kept))
 
    return cleaned
 
 
# ==========================================
# PDF DOWNLOAD (optional)
# ==========================================
 
def download_pdf(pdf_url):
    name = unquote(pdf_url.split("/")[-1].split("?")[0])
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    path = PDF_FOLDER / name
 
    if path.exists():
        return
 
    try:
        response = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            path.write_bytes(response.content)
            print("PDF saved:", name)
        else:
            print("PDF failed:", response.status_code, pdf_url)
    except Exception as error:
        print("PDF error:", error)
 
 
# ==========================================
# MAIN
# ==========================================
 
def main():
    DATA_FOLDER.mkdir(exist_ok=True)
    PDF_FOLDER.mkdir(exist_ok=True)
 
    to_visit = [normalize_url(START_URL), normalize_url(START_URL + "/vision")]
    visited = set()
    pdf_urls = set()
    pages = []
 
    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue
 
        try:
            print(f"[{len(visited) + 1}/{MAX_PAGES}] {url}")
            response = requests.get(url, timeout=20, headers=HEADERS,
                                    allow_redirects=True)
 
            if response.status_code != 200:
                print("   failed:", response.status_code)
                visited.add(url)
                continue
 
            if "text/html" not in response.headers.get("Content-Type", ""):
                visited.add(url)
                continue
 
            visited.add(url)
 
            # ---- find links (before cleaning removes the menus) ----
            link_soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")
            for a in link_soup.find_all("a", href=True):
                target = urljoin(url, a["href"].strip())
                if ".pdf" in target.lower():
                    pdf_urls.add(target.split("#")[0])
                    continue
                target = normalize_url(target)
                if wanted(target) and target not in visited and target not in to_visit:
                    to_visit.append(target)
 
            # ---- clean the page text ----
            title, lines = page_to_lines(response.content)
            pages.append((url, title, lines))
 
            time.sleep(DELAY_SECONDS)
 
        except Exception as error:
            print("   error:", error)
 
    # ---- remove menus/banners that repeat on many pages ----
    pages = remove_repeated_lines(pages)
 
    output = []
    dropped = []
    kept_pages = 0
    for url, title, lines in pages:
        body = "\n".join(lines)
        if len(body) < MIN_PAGE_CHARS:   # empty / redirect / near-empty page
            dropped.append((url, len(body)))
            continue
        output.append(f"SOURCE: {url}\n{title}\n{body}")
        kept_pages += 1
 
    (DATA_FOLDER / "website.txt").write_text("\n\n".join(output), encoding="utf-8")
 
    print("\n" + "=" * 60)
    print("CRAWLING COMPLETED")
    print("=" * 60)
    print("Pages visited:", len(visited))
    print("Pages saved  :", kept_pages)
    print("PDF links    :", len(pdf_urls))
 
    if dropped:
        print(f"\nDROPPED {len(dropped)} pages with almost no text left (url, characters):")
        for url, size in dropped:
            print(f"   {size:4d}  {url}")
 
    if DOWNLOAD_PDFS:
        for pdf_url in pdf_urls:
            download_pdf(pdf_url)
 
    # ---- quick visual check: the menu text must be GONE ----
    print("\nPREVIEW of first saved page (should start with real content, not the menu):")
    print("-" * 60)
    if output:
        print(output[0][:600])
 
 
if __name__ == "__main__":
    main()
 