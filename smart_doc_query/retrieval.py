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
- Page 5: "apply Natural Language Processing techniques to analyze and
  process both document content and user queries effectively"
- Page 9: "standard NLP and machine learning libraries"

This module implements the described behaviour using:
1. sentence-transformers for vector embeddings (semantic similarity)
2. cosine similarity for ML-based similarity matching
3. transformers NLP pipeline for query analysis and answer generation
"""

import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

import database

# ── Model 1: Embedding model for semantic similarity ─────────────────────
# Study Phase (Page 4): "corresponding vector representations"
# Study Phase (Page 5): "machine learning-based similarity matching"
# Downloads ~80MB on first run, cached after
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ── Model 2: NLP model for answer extraction ─────────────────────────────
# Study Phase (Page 4): "analyzes the query using NLP techniques"
# Study Phase (Page 4): "generates an accurate answer strictly based
#   on the retrieved content"
# Study Phase (Page 5): "apply NLP techniques to analyze and process
#   both document content and user queries effectively"
# Downloads ~250MB on first run, cached after
nlp_answering = pipeline(
    'question-answering',
    model='distilbert-base-cased-distilled-squad',
    tokenizer='distilbert-base-cased-distilled-squad'
)

NOT_FOUND_MSG = 'The uploaded documents do not contain enough information to answer this question.'


# ── Vector embedding functions ────────────────────────────────────────────
# Study Phase (Page 4): "stored in a database along with their
# corresponding vector representations for efficient retrieval"

def embed_to_blob(text):
    """Encode text to a vector and serialize for database storage."""
    vector = embedding_model.encode(text)
    return pickle.dumps(vector)


def blob_to_array(blob):
    """Deserialize a stored vector back to a numpy array."""
    return pickle.loads(blob)


# ── Semantic similarity retrieval ─────────────────────────────────────────
# Study Phase (Page 4): "retrieves the most relevant document sections
# based on semantic similarity"

def get_relevant_sections(query_text, top_k=5):
    """Retrieve the most relevant document sections using semantic similarity.

    Study Phase (Page 4): "ensures that related information is identified
    even if different words or sentence structures are used"
    """
    query_vec = embedding_model.encode(query_text).reshape(1, -1)

    rows = database.query_db(
        'SELECT section_id, document_id, section_text, embedding FROM DOCUMENT_SECTION'
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if row['embedding'] is None:
            section_vec = embedding_model.encode(row['section_text']).reshape(1, -1)
        else:
            section_vec = blob_to_array(row['embedding']).reshape(1, -1)

        score = float(cosine_similarity(query_vec, section_vec)[0][0])
        results.append({
            'section_id':   row['section_id'],
            'document_id':  row['document_id'],
            'section_text': row['section_text'],
            'score':        score
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate near-identical chunks
    seen = set()
    deduplicated = []
    for r in results:
        fingerprint = re.sub(r'\s+', ' ', r['section_text'][:100]).strip().lower()
        fingerprint = re.sub(r'^\d{1,3}\s+', '', fingerprint)
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduplicated.append(r)

    # Filter out low-relevance sections
    filtered = [r for r in deduplicated if r['score'] >= 0.25]

    return filtered[:top_k]


# ── Text cleaning ─────────────────────────────────────────────────────────

def _clean_text(text):
    """Clean text for display — remove PDF artifacts."""
    text = re.sub(r'^\d{1,3}\s+(?=[A-Z])', '', text.strip())
    text = re.sub(r'[●▪▸\u2022\u2023\u2043]', '•', text)
    text = text.replace('□', '•')
    text = re.sub(r'\b(soft|hard|fire|net|data|mal|cyber)\s+(ware|wall|work|base|icious|security)\b',
                  r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()


# ── Answer generation using NLP techniques ────────────────────────────────
# Study Phase (Page 4): "analyzes the query using NLP techniques"
# Study Phase (Page 4): "generates an accurate answer strictly based
#   on the retrieved content"
# Study Phase (Page 4): "This retrieval-augmented approach prevents
#   the system from generating unsupported or irrelevant responses"

def generate_answer(query_text, sections):
    """Generate an accurate answer using NLP techniques on retrieved content.

    The process:
    1. The embedding model retrieves the most relevant document sections
       (semantic similarity — already done before this function is called)
    2. The NLP model analyzes the query and extracts the precise answer
       from the retrieved content

    This ensures the answer is:
    - Accurate (NLP model understands what is being asked)
    - Strictly based on retrieved content (can only answer from given text)
    - Not unsupported or irrelevant (extracted, not generated freely)
    """
    if not sections:
        return NOT_FOUND_MSG

    # Build context from top retrieved sections (cleaned)
    # Use top 3 sections to give the NLP model enough content to work with
    context_parts = []
    total_words = 0
    for section in sections[:3]:
        cleaned = _clean_text(section['section_text'])
        words = len(cleaned.split())
        if total_words + words > 800:
            break
        context_parts.append(cleaned)
        total_words += words

    context = ' '.join(context_parts)

    if not context or len(context) < 20:
        return NOT_FOUND_MSG

    # Use NLP techniques to analyze the query and extract the answer
    try:
        result = nlp_answering(question=query_text, context=context)
        answer = result['answer'].strip()
        confidence = result['score']

        # If NLP model is confident, return its focused answer
        # If confidence is low, fall back to the full retrieved section
        if confidence >= 0.01 and len(answer) > 10:
            return _clean_text(answer)
        else:
            return _clean_text(sections[0]['section_text'])

    except Exception:
        # Fallback: return the most relevant section directly
        return _clean_text(sections[0]['section_text'])
