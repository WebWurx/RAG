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
    # Remove replacement characters
    text = text.replace('\ufffd', '')

    # Normalize bullet characters to standard bullet
    text = text.replace('□', '•')
    text = re.sub(r'[●▪▸]', '•', text)

    # Remove standalone page numbers (a line that is just a number)
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)

    # Fix multiple spaces into single space
    text = re.sub(r'  +', ' ', text)

    # Fix space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Fix common PDF word breaks
    text = re.sub(r'\b(soft|hard|fire|net|data|mal|cyber)\s+(ware|wall|work|base|icious|security)\b',
                  r'\1\2', text, flags=re.IGNORECASE)

    return text.strip()


def extract_text_from_pdf(filepath):
    """Extract full text from a PDF file."""
    text = ''
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return _clean_text(text)


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
