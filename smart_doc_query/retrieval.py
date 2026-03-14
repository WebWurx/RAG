import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import database


def embed_to_blob(text):
    """Store raw text as blob — TF-IDF is computed at query time."""
    return pickle.dumps(text)


def blob_to_text(blob):
    return pickle.loads(blob)


def get_relevant_sections(query_text, top_k=5):
    rows = database.query_db(
        'SELECT section_id, document_id, section_text FROM DOCUMENT_SECTION'
    )
    if not rows:
        return []

    texts = [row['section_text'] for row in rows]
    all_texts = texts + [query_text]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vec = tfidf_matrix[-1]
    section_vecs = tfidf_matrix[:-1]

    scores = cosine_similarity(query_vec, section_vecs)[0]

    results = []
    for i, row in enumerate(rows):
        results.append({
            'section_id': row['section_id'],
            'document_id': row['document_id'],
            'section_text': row['section_text'],
            'score': float(scores[i])
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


RELEVANCE_THRESHOLD = 0.05  # minimum score for the top section to be considered relevant

def generate_answer(query_text, sections):
    if not sections:
        return 'The document does not contain enough information to answer this question.'

    # If the best-matching section is below the relevance threshold, the query is off-topic
    if sections[0]['score'] < RELEVANCE_THRESHOLD:
        return 'The document does not contain enough information to answer this question.'

    # Collect sentences only from sections that are meaningfully relevant
    all_sentences = []
    for rank, section in enumerate(sections):
        if section['score'] < RELEVANCE_THRESHOLD:
            continue
        sentences = [s.strip() for s in section['section_text'].replace('\n', ' ').split('.') if len(s.strip()) > 20]
        for pos, sentence in enumerate(sentences):
            all_sentences.append((rank, pos, sentence))

    if not all_sentences:
        return 'The document does not contain enough information to answer this question.'

    # Score sentences by TF-IDF similarity to the query
    all_texts = [s for _, _, s in all_sentences] + [query_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return 'The document does not contain enough information to answer this question.'

    query_vec = tfidf_matrix[-1]
    sent_vecs = tfidf_matrix[:-1]
    tfidf_scores = cosine_similarity(query_vec, sent_vecs)[0]

    # Combine TF-IDF score with a positional bonus so early sentences in top sections surface
    ranked = []
    for i, (rank, pos, sentence) in enumerate(all_sentences):
        position_bonus = 0.15 if (rank == 0 and pos < 3) else 0.0
        final_score = tfidf_scores[i] + position_bonus
        ranked.append((final_score, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in ranked[:5] if s.strip()]

    if not top_sentences:
        return 'The document does not contain enough information to answer this question.'

    answer = '. '.join(top_sentences)
    if not answer.endswith('.'):
        answer += '.'
    return answer
