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

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import database

# Load embedding model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text):
    """Generate vector representation for a text string.

    Study Phase (Page 4): "corresponding vector representations"
    """
    return model.encode(text)


def get_relevant_sections(query_text, top_k=5):
    """Retrieve the most relevant document sections based on semantic similarity.

    Study Phase (Page 4): "retrieves the most relevant document sections
    based on semantic similarity"
    """
    query_vec = model.encode(query_text).reshape(1, -1)

    rows = database.query_db(
        'SELECT section_id, document_id, section_text FROM DOCUMENT_SECTION'
    )
    if not rows:
        return []

    # Compute similarity between query and each stored section
    results = []
    for row in rows:
        section_vec = model.encode(row['section_text']).reshape(1, -1)
        score = float(cosine_similarity(query_vec, section_vec)[0][0])
        results.append({
            'section_id':   row['section_id'],
            'document_id':  row['document_id'],
            'section_text': row['section_text'],
            'score':        score
        })

    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:top_k]


def generate_answer(query_text, sections):
    """Generate an accurate answer strictly based on the retrieved content.

    Study Phase (Page 4): "generates an accurate answer strictly based
    on the retrieved content"
    Study Phase (Page 4): "This retrieval-augmented approach prevents
    the system from generating unsupported or irrelevant responses"
    """
    if not sections:
        return 'The uploaded documents do not contain enough information to answer this question.'

    # Use the most relevant section as the answer
    # The answer is extracted directly from document content (not generated freely)
    best_section = sections[0]['section_text']
    return best_section
