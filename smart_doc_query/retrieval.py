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

    # Deduplicate: skip chunks that start with the same text
    seen = set()
    deduplicated = []
    for r in results:
        fingerprint = r['section_text'][:80].strip().lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduplicated.append(r)

    # Filter out low-relevance sections
    filtered = [r for r in deduplicated if r['score'] >= 0.25]

    return filtered[:top_k]


def generate_answer(query_text, sections):
    """Generate an accurate answer strictly based on the retrieved content.

    Study Phase (Page 4): "generates an accurate answer strictly based
    on the retrieved content"
    Study Phase (Page 4): "This retrieval-augmented approach prevents
    the system from generating unsupported or irrelevant responses"

    Improvement: Instead of dumping the entire top chunk, this extracts
    individual sentences from all retrieved sections, scores each one
    against the query, and returns only the most relevant sentences
    as a focused answer.
    """
    if not sections:
        return NOT_FOUND_MSG

    query_vec = model.encode(query_text).reshape(1, -1)

    # Collect all sentences from retrieved sections
    all_sentences = []
    for section in sections:
        # Split section text into sentences
        fragments = re.split(r'(?<=[.?!])\s+', section['section_text'].replace('\n', ' '))
        for frag in fragments:
            cleaned = frag.strip()
            # Only keep sentences that are long enough to be meaningful
            if len(cleaned) > 20:
                all_sentences.append(cleaned)

    if not all_sentences:
        return NOT_FOUND_MSG

    # Score each sentence against the query using semantic similarity
    sentence_vecs = model.encode(all_sentences)
    scores = cosine_similarity(query_vec, sentence_vecs)[0]

    # Rank sentences by score
    ranked = sorted(zip(scores, all_sentences), key=lambda x: x[0], reverse=True)

    # Filter: only keep sentences scoring at least 25% of the top score
    top_score = ranked[0][0] if ranked else 0
    min_score = top_score * 0.25

    # Pick top relevant sentences, deduplicated
    seen_text = set()
    answer_sentences = []
    for score, sentence in ranked:
        if score < min_score:
            break
        normalized = sentence.strip().lower()
        if normalized not in seen_text:
            seen_text.add(normalized)
            answer_sentences.append(sentence.strip())
        if len(answer_sentences) == 6:
            break

    if not answer_sentences:
        return NOT_FOUND_MSG

    return '\n\n'.join(answer_sentences)
