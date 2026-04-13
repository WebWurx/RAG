"""
Retrieval Module

Study Phase references:
- Page 4: "stored in a database along with their corresponding vector
  representations for efficient retrieval"
- Page 4: "analyzes the query using Natural Language Processing techniques"
- Page 4: "retrieves the most relevant document sections based on
  semantic similarity"
- Page 4: "generates an accurate answer strictly based on the retrieved content"
- Page 5: "machine learning-based similarity matching"

The docs do not specify which embedding model, similarity algorithm,
or answer generation method to use. This module implements the described
behaviour using sentence-transformers for vector embeddings and
cosine similarity for semantic matching.
"""

import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import database

# Load embedding model once at startup — downloads ~80MB on first run, cached after
model = SentenceTransformer('all-MiniLM-L6-v2')

NOT_FOUND_MSG = 'The uploaded documents do not contain enough information to answer this question.'


# ── Improvement 2: Store embeddings at upload time ────────────────────────
# Study Phase (Page 4): "stored in a database along with their
# corresponding vector representations for efficient retrieval"
#
# Instead of re-encoding every section on every query, we encode once
# at upload time and store the vector as a pickle blob in the DB.
# This requires adding an 'embedding' column to DOCUMENT_SECTION —
# done via migration in database.py.


def embed_to_blob(text):
    """Encode text to a vector and serialize for database storage.

    Study Phase (Page 4): "corresponding vector representations"
    """
    vector = model.encode(text)
    return pickle.dumps(vector)


def blob_to_array(blob):
    """Deserialize a stored vector back to a numpy array."""
    return pickle.loads(blob)


def get_relevant_sections(query_text, top_k=5):
    """Retrieve the most relevant document sections based on semantic similarity.

    Study Phase (Page 4): "retrieves the most relevant document sections
    based on semantic similarity"
    Study Phase (Page 4): "ensures that related information is identified
    even if different words or sentence structures are used"

    Improvements over basic version:
    - Uses pre-stored embeddings (no re-encoding on every query)
    - Filters out low-relevance sections (threshold)
    - Deduplicates near-identical chunks
    """
    query_vec = model.encode(query_text).reshape(1, -1)

    rows = database.query_db(
        'SELECT section_id, document_id, section_text, embedding FROM DOCUMENT_SECTION'
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if row['embedding'] is None:
            # Fallback: encode on the fly if no stored embedding
            section_vec = model.encode(row['section_text']).reshape(1, -1)
        else:
            section_vec = blob_to_array(row['embedding']).reshape(1, -1)

        score = float(cosine_similarity(query_vec, section_vec)[0][0])
        results.append({
            'section_id':   row['section_id'],
            'document_id':  row['document_id'],
            'section_text': row['section_text'],
            'score':        score
        })

    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate: skip chunks with similar content
    seen = set()
    deduplicated = []
    for r in results:
        # Normalize: strip page numbers, collapse whitespace, lowercase
        fingerprint = re.sub(r'\s+', ' ', r['section_text'][:100]).strip().lower()
        fingerprint = re.sub(r'^\d{1,3}\s+', '', fingerprint)
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduplicated.append(r)

    # Filter out low-relevance sections
    filtered = [r for r in deduplicated if r['score'] >= 0.25]

    return filtered[:top_k]


def _clean_chunk(text):
    """Clean a chunk for display in the answer.

    Removes page numbers, normalizes bullets and whitespace,
    fixes common PDF extraction artifacts.
    """
    # Remove leading page numbers
    text = re.sub(r'^\d{1,3}\s+(?=[A-Z])', '', text.strip())

    # Normalize bullet characters to a clean format
    text = re.sub(r'[●▪▸\u2022\u2023\u2043]', '•', text)
    text = text.replace('□', '•')

    # Fix common PDF word breaks
    text = re.sub(r'\b(soft|hard|fire|net|data|mal|cyber)\s+(ware|wall|work|base|icious|security)\b',
                  r'\1\2', text, flags=re.IGNORECASE)

    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)

    # Fix space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    return text.strip()


def _normalize_for_dedup(text):
    """Normalize text for deduplication — aggressive stripping."""
    t = text.lower().strip()
    t = re.sub(r'^\d{1,3}\s+', '', t)
    t = re.sub(r'^[•●▪▸\-\*]\s*', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def generate_answer(query_text, sections):
    """Generate an accurate answer strictly based on the retrieved content.

    Study Phase (Page 4): "generates an accurate answer strictly based
    on the retrieved content"
    Study Phase (Page 4): "This retrieval-augmented approach prevents
    the system from generating unsupported or irrelevant responses"

    Strategy: Return the top 2-3 most relevant chunks directly, cleaned
    and deduplicated. The chunking already creates meaningful sections —
    if chunks are good, they ARE the answer. This works for lists, prose,
    definitions, and any PDF format without fragile regex parsing.
    """
    if not sections:
        return NOT_FOUND_MSG

    # Clean and deduplicate the retrieved sections
    seen = set()
    clean_sections = []
    for section in sections:
        cleaned = _clean_chunk(section['section_text'])
        norm = _normalize_for_dedup(cleaned[:100])
        if norm not in seen and len(cleaned) > 20:
            seen.add(norm)
            clean_sections.append(cleaned)

    if not clean_sections:
        return NOT_FOUND_MSG

    # Return top 2-3 chunks — trimmed to ~500 words max total
    answer_parts = []
    total_words = 0
    max_words = 500

    for chunk in clean_sections[:3]:
        words = chunk.split()
        if total_words + len(words) > max_words and answer_parts:
            break
        answer_parts.append(chunk)
        total_words += len(words)

    return '\n\n'.join(answer_parts)
