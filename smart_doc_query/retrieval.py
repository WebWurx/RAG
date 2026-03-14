import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import database

RELEVANCE_THRESHOLD = 0.15  # drop any chunk below 15% similarity
NOT_FOUND_MSG = 'The document does not contain enough information to answer this question.'

# Short vague queries benefit from expansion so TF-IDF has more signal to work with
QUERY_EXPANSIONS = {
    'what is this project':  'what is the purpose description and overview of this project system',
    'what is the project':   'what is the purpose description and overview of this project system',
    'what is the system':    'what is the purpose description and overview of this system',
    'who made this':         'who is the author submitted by name of this project',
    'who did this':          'who is the author submitted by name of this project',
    'who did this project':  'who is the author submitted by name of this project',
    'what are the features': 'what are the features functions and capabilities of the system',
    'what does it do':       'what is the purpose function and capability of the system',
}


def expand_query(query_text):
    key = query_text.lower().strip().rstrip('?')
    return QUERY_EXPANSIONS.get(key, query_text)


def embed_to_blob(text):
    """Store raw text as blob — TF-IDF is computed at query time."""
    return pickle.dumps(text)


def blob_to_text(blob):
    return pickle.loads(blob)


def get_relevant_sections(query_text, top_k=5):
    query_text = expand_query(query_text)

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
            'section_id':   row['section_id'],
            'document_id':  row['document_id'],
            'section_text': row['section_text'],
            'score':        float(scores[i])
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    # --- Deduplication: keep only the highest-scoring chunk per unique text fingerprint ---
    seen_fingerprints = set()
    deduplicated = []
    for r in results:
        # Use first 80 chars as a near-duplicate fingerprint
        fingerprint = r['section_text'][:80].strip().lower()
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            deduplicated.append(r)

    # --- Threshold filter: drop chunks below 15% similarity ---
    filtered = [r for r in deduplicated if r['score'] >= RELEVANCE_THRESHOLD]

    return filtered[:top_k]


def generate_answer(query_text, sections):
    if not sections:
        return NOT_FOUND_MSG

    # Collect sentences only from sections that passed the threshold
    all_sentences = []
    for rank, section in enumerate(sections):
        sentences = [
            s.strip()
            for s in section['section_text'].replace('\n', ' ').split('.')
            if len(s.strip()) > 20
        ]
        for pos, sentence in enumerate(sentences):
            all_sentences.append((rank, pos, sentence))

    if not all_sentences:
        return NOT_FOUND_MSG

    # Score sentences by TF-IDF similarity to the (possibly expanded) query
    expanded_query = expand_query(query_text)
    all_texts = [s for _, _, s in all_sentences] + [expanded_query]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return NOT_FOUND_MSG

    query_vec = tfidf_matrix[-1]
    sent_vecs = tfidf_matrix[:-1]
    tfidf_scores = cosine_similarity(query_vec, sent_vecs)[0]

    # Combine TF-IDF score with a positional bonus for early sentences in the top section
    ranked = []
    for i, (rank, pos, sentence) in enumerate(all_sentences):
        position_bonus = 0.15 if (rank == 0 and pos < 3) else 0.0
        ranked.append((tfidf_scores[i] + position_bonus, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate sentences in the answer
    seen = set()
    top_sentences = []
    for _, sentence in ranked:
        norm = sentence.strip().lower()
        if norm not in seen and sentence.strip():
            seen.add(norm)
            top_sentences.append(sentence.strip())
        if len(top_sentences) == 5:
            break

    if not top_sentences:
        return NOT_FOUND_MSG

    answer = '. '.join(top_sentences)
    if not answer.endswith('.'):
        answer += '.'
    return answer
