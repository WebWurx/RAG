"""
Document Processor Module

Study Phase (Page 4):
- "The uploaded documents are processed to extract textual content"
- "which is then divided into smaller meaningful sections"

No specific chunking strategy, chunk size, or overlap is mentioned in the docs.
Using a simple word-based splitting approach to create meaningful sections.
"""

import PyPDF2


def extract_text_from_pdf(filepath):
    """Extract full text from a PDF file."""
    text = ''
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text.strip()


def extract_text_from_txt(filepath):
    """Extract full text from a TXT file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().strip()


def chunk_text(text):
    """
    Divide text into smaller meaningful sections.

    Study Phase (Page 4): "divided into smaller meaningful sections"
    No chunk size or overlap strategy is specified in the documentation.
    """
    words = text.split()
    chunks = []
    chunk_size = 150  # words per section
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks
