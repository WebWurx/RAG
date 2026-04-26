"""
Document Processor Module

Study Phase (Page 4):
- "The uploaded documents are processed to extract textual content"
- "which is then divided into smaller meaningful sections"

No specific chunking strategy, chunk size, or overlap is mentioned in the docs.
Using a simple word-based splitting approach to create meaningful sections.
"""

import PyPDF2
import re


def _clean_text(text):
    """Clean extracted PDF text.

    Removes standalone page numbers, fixes broken spacing from PDF
    extraction, normalizes bullet characters, and strips junk.
    """
    if not text:
        return ''

    # Remove replacement characters
    text = text.replace('\ufffd', '')

    # Normalize weird bullet characters to standard bullet + space
    text = re.sub(r'[▯□●▪▸❖❑✓✗\u2022\u2023\u2043\uf0a7\uf0b7\uf0a8\uf0fc\uf0a4\uf0d8]', '• ', text)

    # Remove standalone page numbers (a line that is just a number)
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)

    # Remove "(B)" / "(A)" style site markers that appear mid-sentence
    text = re.sub(r'\s*\([AB]\)\s*', ' ', text)

    # Fix known broken words with spaces inside
    broken_word_fixes = {
        r'\bat t he\b': 'at the',
        r'\balg orithm\b': 'algorithm',
        r'\be -mail\b': 'email',
        r'\bradix -64\b': 'radix-64',
        r'\bkey -exchange\b': 'key-exchange',
        r'\bPhase -1\b': 'Phase-1',
        r'\bPhase -2\b': 'Phase-2',
        r'\bChange -cipher\b': 'Change-cipher',
        r'\bnon -threatening\b': 'non-threatening',
        r'\bmalici ous\b': 'malicious',
    }
    for pattern, replacement in broken_word_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Fix common PDF word breaks (including "soft ware" → "software", etc.)
    text = re.sub(
        r'\b(soft|hard|fire|net|data|mal|cyber)\s+(ware|wall|work|base|icious|security)\b',
        r'\1\2', text, flags=re.IGNORECASE
    )

    # Remove extra spaces around hyphens: "word - word" → "word-word"
    text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    text = re.sub(r'(\w)\s+-(\w)', r'\1-\2', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Fix space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    return text.strip()


def extract_text_from_pdf(filepath):
    """Extract full text from a PDF file (joined). Kept for backward compat."""
    pages = extract_pages_from_pdf(filepath)
    return _clean_text('\n'.join(pt for _, pt in pages))


def extract_pages_from_pdf(filepath):
    """Extract text per-page. Returns [(page_number, cleaned_text), …].

    Page numbers are 1-indexed to match the human-readable convention used in
    the "Answer based on: doc · p. 3" UI.
    """
    pages = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ''
            cleaned = _clean_text(page_text)
            if cleaned.strip():
                pages.append((i, cleaned))
    return pages


def extract_text_from_txt(filepath):
    """Extract full text from a TXT file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return _clean_text(f.read())


def _split_sentences(text):
    """Split text into sentences using common delimiters."""
    import re
    # Split on period/question/exclamation followed by space or newline
    raw = re.split(r'(?<=[.?!])\s+', text.replace('\n', ' '))
    return [s.strip() for s in raw if len(s.strip()) > 10]


def chunk_text(text):
    """
    Divide text into smaller meaningful sections with overlap.

    Study Phase (Page 4): "divided into smaller meaningful sections"

    Sentence-aware chunking ensures no sentence is split mid-way.
    Overlap between chunks preserves context at boundaries.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    chunk_size = 150   # target words per section
    overlap = 50       # overlap words between sections
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence exceeds chunk size and we have content,
        # save current chunk and start new one with overlap
        if current_words + word_count > chunk_size and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            chunks.append(chunk_text_str.strip())

            # Build overlap: take last few sentences that fit within overlap words
            overlap_sentences = []
            overlap_words = 0
            for s in reversed(current_chunk):
                s_words = len(s.split())
                if overlap_words + s_words <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_words += s_words
                else:
                    break

            current_chunk = overlap_sentences
            current_words = overlap_words

        current_chunk.append(sentence)
        current_words += word_count

    # Don't forget the last chunk
    if current_chunk:
        chunk_text_str = ' '.join(current_chunk)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str.strip())

    return chunks


def chunk_pages(pages):
    """Page-aware chunking. Each chunk belongs to exactly one page so the
    "Answer based on: doc · p. 3" reference is unambiguous.

    `pages` is a list of (page_number, page_text) tuples.
    Returns a list of (page_number, chunk_text) tuples.
    """
    out = []
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            out.append((page_num, chunk))
    return out
