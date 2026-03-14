import PyPDF2


def extract_text_from_pdf(filepath):
    """Returns list of (page_number, text) tuples, one per page."""
    pages = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i, text.strip()))
    return pages


def extract_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().strip()


def chunk_pdf_pages(pages, chunk_size=175, overlap=25):
    """
    Chunk PDF text while keeping track of the source page number.
    Returns list of (chunk_text, page_number) tuples.
    """
    chunks = []
    step = chunk_size - overlap
    for page_num, text in pages:
        words = text.split()
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append((chunk.strip(), page_num))
    return chunks


def chunk_text(text, chunk_size=175, overlap=25):
    """Chunk plain text (TXT files). Returns list of (chunk_text, page_number=1)."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append((chunk.strip(), 1))
    return chunks
