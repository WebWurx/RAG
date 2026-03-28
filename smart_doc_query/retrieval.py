import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import database

# Load model once at startup — downloads ~80MB on first run, cached after
model = SentenceTransformer('all-MiniLM-L6-v2')

RELEVANCE_THRESHOLD = 0.25
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

# Number words recognised in list questions
_COUNT_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}

# Generic nouns that signal the user wants a list
_LIST_NOUN_PATTERN = (
    r'(aspects?|types?|parts?|components?|steps?|methods?|phases?|elements?|'
    r'categories?|features?|principles?|properties?|characteristics?|factors?|'
    r'stages?|functions?|layers?|levels?|modes?|forms?|kinds?|ways?|points?|'
    r'items?|examples?|reasons?|advantages?|disadvantages?|goals?|objectives?)'
)


def detect_question_type(query_text):
    """
    Classify query as 'definition', 'list', or 'general' purely from grammar.
    No topic-specific knowledge used.
    """
    q = query_text.lower().strip().rstrip('?')

    # List: explicit count word or digit before a list noun
    count_words = '|'.join(_COUNT_WORDS.keys())
    if re.search(rf'\b({count_words}|\d+)\b.*{_LIST_NOUN_PATTERN}', q):
        return 'list'

    # List: "what are the <list noun> of ..."
    if re.search(rf'\bwhat are the\b.*{_LIST_NOUN_PATTERN}', q):
        return 'list'

    # List: starts with list/name/enumerate
    if re.match(r'^(list|name|enumerate)\b', q):
        return 'list'

    # Definition: "what is ...", "what are ..." (no list noun already matched)
    if re.match(r'^what (is|are)\b', q):
        return 'definition'

    # Definition: "define ...", "what does ... mean", "what is meant by ..."
    if re.match(r'^define\b', q):
        return 'definition'
    if re.search(r'what (is|does).+mean', q):
        return 'definition'
    if re.match(r'^what is meant by\b', q):
        return 'definition'
    if re.match(r'^explain what\b', q):
        return 'definition'

    return 'general'


def extract_list_count(query_text):
    """Return the integer count requested in a list question, or None."""
    q = query_text.lower()
    for word, n in _COUNT_WORDS.items():
        if re.search(rf'\b{word}\b', q):
            return n
    m = re.search(r'\b(\d+)\b', q)
    if m:
        return int(m.group(1))
    return None


def extract_list_items(section_text):
    """
    Extract list-structured items from a section using text formatting patterns.
    Works on any document — no domain knowledge required.
    """
    items = []

    # Line-based extraction: numbered or bulleted lines
    lines = section_text.split('\n')
    for line in lines:
        line = line.strip()
        # Numbered: "1.", "1)", "(1)", "a.", "a)"
        if re.match(r'^(\d+[\.\)]|\(\d+\)|[a-zA-Z][\.\)])\s+\S', line):
            item = re.sub(r'^(\d+[\.\)]|\(\d+\)|[a-zA-Z][\.\)])\s+', '', line).strip()
            if len(item) >= 15:
                items.append(item)
        # Bullet: "- ", "• ", "* "
        elif re.match(r'^[-•*]\s+\S', line):
            item = re.sub(r'^[-•*]\s+', '', line).strip()
            if len(item) >= 15:
                items.append(item)

    if items:
        return items

    # Fallback: sentence-level ordinal detection ("First, ...", "Second, ...")
    sentences = [s.strip() for s in section_text.replace('\n', ' ').split('.') if len(s.strip()) >= 15]
    ordinals = r'^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)[,:\s]'
    for s in sentences:
        if re.match(ordinals, s.lower()):
            items.append(s)

    return items


def expand_query(query_text):
    key = query_text.lower().strip().rstrip('?')
    return QUERY_EXPANSIONS.get(key, query_text)


def embed_to_blob(text):
    vector = model.encode(text)
    return pickle.dumps(vector)


def blob_to_array(blob):
    return pickle.loads(blob)


def get_relevant_sections(query_text, top_k=8):
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

    q_type = detect_question_type(query_text)
    expanded = expand_query(query_text)
    query_vec = model.encode(expanded).reshape(1, -1)

    # ── LIST PATH ─────────────────────────────────────────────────────────────
    if q_type == 'list':
        n = extract_list_count(query_text)

        all_items = []
        for section in sections:
            all_items.extend(extract_list_items(section['section_text']))

        if all_items:
            # Deduplicate
            seen = set()
            unique_items = []
            for item in all_items:
                norm = item.strip().lower()
                if norm not in seen:
                    seen.add(norm)
                    unique_items.append(item.strip())

            # Score items against query using sentence-transformers + cosine similarity
            item_vecs = model.encode(unique_items)
            scores = cosine_similarity(query_vec, item_vecs)[0]
            ranked_items = sorted(zip(scores, unique_items), key=lambda x: x[0], reverse=True)

            limit = n if n else len(ranked_items)
            top_items = [item for _, item in ranked_items[:limit]]

            if top_items:
                return '\n\n'.join(top_items)
        # No structured list found — fall through to sentence extraction

    # ── COLLECT SENTENCES (shared by definition + general paths) ──────────────
    all_sentences = []
    for rank, section in enumerate(sections):
        fragments = re.split(r'[.\n•■▌]+', section['section_text'])
        sentences = []
        for frag in fragments:
            cleaned = re.sub(r'^\s*\d+\s*', '', frag)
            cleaned = re.sub(r'^[•■▌\-\*\s]+', '', cleaned).strip()
            if len(cleaned) > 20:
                sentences.append(cleaned)
        for pos, sentence in enumerate(sentences):
            all_sentences.append((rank, pos, sentence))

    if not all_sentences:
        return NOT_FOUND_MSG

    sentence_texts = [s for _, _, s in all_sentences]
    sentence_vecs = model.encode(sentence_texts)
    scores = cosine_similarity(query_vec, sentence_vecs)[0]

    # ── DEFINITION PATH ───────────────────────────────────────────────────────
    if q_type == 'definition':
        ranked = [
            (float(scores[i]), sentence)
            for i, (_, _, sentence) in enumerate(all_sentences)
        ]
        ranked.sort(key=lambda x: x[0], reverse=True)
        top_score = ranked[0][0] if ranked else 0
        min_score = top_score * 0.70

        for score, sentence in ranked:
            if score < min_score:
                break
            if sentence.strip():
                return sentence.strip()

        return NOT_FOUND_MSG

    # ── GENERAL PATH ──────────────────────────────────────────────────────────
    ranked = []
    for i, (rank, pos, sentence) in enumerate(all_sentences):
        position_bonus = 0.05 if (rank == 0 and pos < 3) else 0.0
        ranked.append((float(scores[i]) + position_bonus, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)

    top_score = ranked[0][0] if ranked else 0
    min_score = top_score * 0.25

    seen_text = set()
    top_sentences = []
    for score, sentence in ranked:
        if score < min_score:
            break
        norm = sentence.strip().lower()
        if norm not in seen_text and sentence.strip():
            seen_text.add(norm)
            top_sentences.append(sentence.strip())
        if len(top_sentences) == 6:
            break

    if not top_sentences:
        return NOT_FOUND_MSG

    return '\n\n'.join(list(dict.fromkeys(top_sentences)))
