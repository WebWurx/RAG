import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import database

# Load model once at startup — downloads ~80MB on first run, cached after
model = SentenceTransformer('all-MiniLM-L6-v2')

RELEVANCE_THRESHOLD = 0.25  # neural embeddings score higher than TF-IDF; 25% is a meaningful match
NOT_FOUND_MSG = 'The document does not contain enough information to answer this question.'

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
    vector = model.encode(text)
    return pickle.dumps(vector)


def blob_to_array(blob):
    return pickle.loads(blob)


def get_relevant_sections(query_text, top_k=5):
    expanded = expand_query(query_text)
    query_vec = model.encode(expanded).reshape(1, -1)

    rows = database.query_db(
        'SELECT section_id, document_id, section_text, embedding FROM DOCUMENT_SECTION'
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if row['embedding'] is None:
            continue
        section_vec = blob_to_array(row['embedding']).reshape(1, -1)
        score = float(cosine_similarity(query_vec, section_vec)[0][0])
        results.append({
            'section_id':   row['section_id'],
            'document_id':  row['document_id'],
            'section_text': row['section_text'],
            'score':        score
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    # Deduplication: skip chunks that start the same way
    seen = set()
    deduplicated = []
    for r in results:
        fingerprint = r['section_text'][:80].strip().lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduplicated.append(r)

    # Drop chunks below relevance threshold
    filtered = [r for r in deduplicated if r['score'] >= RELEVANCE_THRESHOLD]

    return filtered[:top_k]


def generate_answer(query_text, sections):
    if not sections:
        return NOT_FOUND_MSG

    expanded = expand_query(query_text)
    query_vec = model.encode(expanded).reshape(1, -1)

    # Collect sentences from all retrieved sections
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

    # Score each sentence using neural embeddings
    sentence_texts = [s for _, _, s in all_sentences]
    sentence_vecs = model.encode(sentence_texts)
    scores = cosine_similarity(query_vec, sentence_vecs)[0]

    # Combine similarity score with positional bonus for early sentences in top section
    ranked = []
    for i, (rank, pos, sentence) in enumerate(all_sentences):
        position_bonus = 0.05 if (rank == 0 and pos < 3) else 0.0
        ranked.append((float(scores[i]) + position_bonus, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # Only keep sentences scoring at least 40% of the top score
    top_score = ranked[0][0] if ranked else 0
    min_score = top_score * 0.4

    seen_text = set()
    top_sentences = []
    for score, sentence in ranked:
        if score < min_score:
            break
        norm = sentence.strip().lower()
        if norm not in seen_text and sentence.strip():
            seen_text.add(norm)
            top_sentences.append(sentence.strip())
        if len(top_sentences) == 3:
            break

    if not top_sentences:
        return NOT_FOUND_MSG

    answer = '. '.join(top_sentences)
    if not answer.endswith('.'):
        answer += '.'
    return answer
