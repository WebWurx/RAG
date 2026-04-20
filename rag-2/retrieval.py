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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

_qa_model_name = 'distilbert-base-cased-distilled-squad'
_qa_tokenizer = AutoTokenizer.from_pretrained(_qa_model_name)
_qa_model = AutoModelForQuestionAnswering.from_pretrained(_qa_model_name)


def _extract_answer(question, context):
    """Use NLP techniques to analyze the query and extract an answer.

    Takes the question and a context passage, returns the most relevant
    answer span extracted from the context. If the extracted span is too
    short (< 15 chars), expand to the full sentence containing it.
    """
    inputs = _qa_tokenizer(question, context, return_tensors='pt',
                           truncation=True, max_length=512)

    with torch.no_grad():
        outputs = _qa_model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    if end_idx < start_idx:
        end_idx = start_idx

    input_ids = inputs['input_ids'][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = _qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    confidence = float(torch.max(start_scores) + torch.max(end_scores))

    answer = answer.strip()

    # If the extracted span is too short, expand it to the full sentence.
    if answer and len(answer) < 15:
        expanded = _expand_to_sentence(answer, context)
        if expanded and len(expanded) > len(answer):
            answer = expanded

    return answer, confidence


def _expand_to_sentence(answer_fragment, context):
    """Given a short answer fragment, find and return the full sentence
    in the original context that contains it. Sentence boundaries are
    ". ", "! ", "? " or start/end of context.
    """
    if not answer_fragment or not context:
        return answer_fragment

    # Locate the fragment in the context (case-insensitive)
    lower_ctx = context.lower()
    lower_frag = answer_fragment.lower()
    pos = lower_ctx.find(lower_frag)
    if pos == -1:
        return answer_fragment

    # Walk backwards to find the start of the sentence
    start = pos
    while start > 0:
        prev = context[start - 1]
        # Look for sentence boundary: ". ", "! ", "? ", or newline
        if prev in '.!?\n' and (start == len(context) or context[start] == ' ' or start < pos):
            break
        start -= 1
    # Skip leading whitespace
    while start < len(context) and context[start] in ' \t\n':
        start += 1

    # Walk forwards to find end of sentence
    end = pos + len(answer_fragment)
    while end < len(context):
        ch = context[end]
        if ch in '.!?':
            end += 1
            break
        end += 1

    sentence = context[start:end].strip()
    return sentence if sentence else answer_fragment

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

_DEFINITION_QUERY_PATTERN = re.compile(
    r'^\s*(what\s+is|what\s+are|what\s+does|define|meaning\s+of|definition\s+of)\b',
    re.IGNORECASE
)


def _is_definition_query(query_text):
    return bool(_DEFINITION_QUERY_PATTERN.match(query_text or ''))


def _extract_key_term(query_text):
    """Extract the key noun/acronym from a definition-style question.

    "What is PGP?" → "PGP"
    "What does SSL stand for?" → "SSL"
    "What is a firewall?" → "firewall"
    """
    if not query_text:
        return ''
    q = query_text.strip().rstrip('?!.').strip()

    # Strip common question leaders + optional leading article
    q = re.sub(
        r'^\s*(what\s+is|what\s+are|what\s+does|define|meaning\s+of|definition\s+of)\s+',
        '', q, flags=re.IGNORECASE
    )
    q = re.sub(r'^\s*(a|an|the)\s+', '', q, flags=re.IGNORECASE)
    # Strip trailing verb phrases like "stand for", "mean"
    q = re.sub(r'\s+(stand\s+for|stands\s+for|mean|means|refer\s+to|refers\s+to).*$',
               '', q, flags=re.IGNORECASE)

    q = q.strip().strip('"\'').strip()

    # Prefer an acronym (all-caps 2-6 letters) if present
    acronym_match = re.search(r'\b([A-Z]{2,6})\b', q)
    if acronym_match:
        return acronym_match.group(1)

    # Otherwise return the first 1-3 significant words
    words = q.split()
    if not words:
        return ''
    # Skip leading articles again just in case
    if words[0].lower() in ('a', 'an', 'the'):
        words = words[1:]
    return ' '.join(words[:3]).strip()


def _definition_boost(section_text, key_term):
    """Return a score boost if section contains a definition pattern
    for the key_term.
    """
    if not key_term or not section_text:
        return 0.0
    term = re.escape(key_term)
    patterns = [
        rf'\b{term}\s+is\b',
        rf'\b{term}\s+are\b',
        rf'\b{term}\s+stands\s+for\b',
        rf'\b{term}\s+refers\s+to\b',
        rf'\b{term}\s+can\s+be\s+defined\s+as\b',
        rf'\b{term}\s+means\b',
    ]
    for p in patterns:
        if re.search(p, section_text, flags=re.IGNORECASE):
            return 0.1
    return 0.0


def get_relevant_sections(query_text, top_k=8):
    """Retrieve the most relevant document sections using semantic similarity.

    Study Phase (Page 4): "ensures that related information is identified
    even if different words or sentence structures are used"

    For definition-style queries ("what is X", "what does X stand for",
    "define X"), sections containing definition patterns like "X is",
    "X stands for", "X refers to" receive a small score boost so the
    actual definition ranks above tangential mentions.
    """
    query_vec = embedding_model.encode(query_text).reshape(1, -1)

    rows = database.query_db(
        'SELECT section_id, document_id, section_text, embedding FROM DOCUMENT_SECTION'
    )
    if not rows:
        return []

    is_def_query = _is_definition_query(query_text)
    key_term = _extract_key_term(query_text) if is_def_query else ''

    results = []
    for row in rows:
        if row['embedding'] is None:
            section_vec = embedding_model.encode(row['section_text']).reshape(1, -1)
        else:
            section_vec = blob_to_array(row['embedding']).reshape(1, -1)

        score = float(cosine_similarity(query_vec, section_vec)[0][0])

        # Apply definition-pattern boost
        if is_def_query and key_term:
            score += _definition_boost(row['section_text'], key_term)

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
    """Clean text for display — remove PDF artifacts.

    Mirrors the cleaning rules in document_processor._clean_text so that
    text stored pre-cleaning and text produced at query time end up with
    the same style.
    """
    if not text:
        return ''

    text = text.replace('\ufffd', '')

    # Remove leading stray section/page number
    text = re.sub(r'^\d{1,3}\s+(?=[A-Z])', '', text.strip())

    # Normalize weird bullet characters
    text = re.sub(r'[▯□●▪▸❖❑✓✗\u2022\u2023\u2043]', '• ', text)

    # Remove "(B)" / "(A)" site markers mid-sentence
    text = re.sub(r'\s*\([AB]\)\s*', ' ', text)

    # Fix known broken words
    broken_word_fixes = {
        r'\bat t he\b': 'at the',
        r'\balg orithm\b': 'algorithm',
        r'\be -mail\b': 'email',
        r'\bradix -64\b': 'radix-64',
        r'\bkey -exchange\b': 'key-exchange',
        r'\bPhase -1\b': 'Phase-1',
        r'\bPhase -2\b': 'Phase-2',
        r'\bChange -cipher\b': 'Change-cipher',
        r'\bnon -threatening\b': 'non-threatening',
        r'\bmalici ous\b': 'malicious',
    }
    for pattern, replacement in broken_word_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(
        r'\b(soft|hard|fire|net|data|mal|cyber)\s+(ware|wall|work|base|icious|security)\b',
        r'\1\2', text, flags=re.IGNORECASE
    )

    # Remove spaces around hyphens between word chars
    text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    text = re.sub(r'(\w)\s+-(\w)', r'\1-\2', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)

    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    return text.strip()


_LIST_QUERY_PATTERN = re.compile(
    r'\b(what\s+are|list|types\s+of|kinds\s+of|examples\s+of)\b',
    re.IGNORECASE
)


def _is_list_query(query_text):
    return bool(_LIST_QUERY_PATTERN.search(query_text or ''))


def _is_header_like(sentence):
    """True when a sentence looks like a header/section title rather
    than real content (mostly uppercase, trailing paren marker, short)."""
    if not sentence:
        return True
    s = sentence.strip()
    if len(s) < 20:
        return True
    # Headers often end with "(A)" / "(B)" style marker
    if re.search(r'\([AB]\)\s*$', s):
        return True
    # Mostly uppercase = header
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.7:
            return True
    return False


def _extract_list_items(text):
    """Extract list items from text — lines starting with numbers,
    bullets, or "•". Returns a list of cleaned item strings.
    """
    if not text:
        return []
    items = []
    # Normalize PDF "o " list artifact to bullet before scanning
    normalized = re.sub(r'(^|\s)o\s+(?=[A-Z])', r'\1• ', text)
    # Split on line breaks first; if none, try splitting before bullet/number
    chunks = re.split(r'(?:\n|(?<=\S)\s+(?=•|\d+[.)]\s))', normalized)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r'^(?:•|[-*]|\d+[.)])\s*(.+)$', chunk)
        if m:
            item = m.group(1).strip().rstrip('.,;:')
            if 2 < len(item) < 200:
                items.append(item)
    return items


def _pick_best_section(query_text, sections, fallback_context):
    """Build an answer from the most relevant retrieved sections.

    - For list-style queries, try to return the extracted list items.
    - Otherwise return the first 3 content sentences of the top section.
    - Skip header-like sentences and very short fragments.
    """
    if not sections:
        return fallback_context

    top_score = sections[0]['score']
    threshold = top_score * 0.80
    relevant = [s for s in sections[:5] if s['score'] >= threshold]
    if not relevant:
        relevant = [sections[0]]

    # For list queries, try to pull out the list items cleanly
    if _is_list_query(query_text):
        for s in relevant:
            text = _clean_text(s['section_text'])
            items = _extract_list_items(text)
            if len(items) >= 2:
                return '\n'.join(f'• {item}' for item in items[:10])

    # Prioritize sections containing numbered lists when relevant
    list_sections = []
    other_sections = []
    for s in relevant:
        text = _clean_text(s['section_text'])
        if re.search(r'\b[1-9][.)]\s', text):
            list_sections.append(text)
        else:
            other_sections.append(text)
    ordered_texts = list_sections + other_sections

    # Otherwise: take the first 3 content sentences from the top ordered text
    parts = []
    seen_sentences = set()
    for text in ordered_texts:
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            sentence = sentence.strip()
            if _is_header_like(sentence):
                continue
            fingerprint = re.sub(r'\s+', ' ', sentence[:80]).lower()
            if fingerprint in seen_sentences:
                continue
            seen_sentences.add(fingerprint)
            parts.append(sentence)
            if len(parts) >= 3:
                break
        if len(parts) >= 3:
            break

    return ' '.join(parts) if parts else fallback_context


# ── Answer generation using NLP techniques ────────────────────────────────
# Study Phase (Page 4): "analyzes the query using NLP techniques"
# Study Phase (Page 4): "generates an accurate answer strictly based
#   on the retrieved content"
# Study Phase (Page 4): "This retrieval-augmented approach prevents
#   the system from generating unsupported or irrelevant responses"

def _polish_answer(answer):
    """Final polish pass applied to every answer before returning it."""
    if not answer:
        return answer

    # Run through the shared cleaning rules one more time
    answer = _clean_text(answer)

    # Replace PDF "o " list-artifact at the start of fragments with "• "
    answer = re.sub(r'(^|\n|\s{2,})o\s+(?=[A-Za-z])', r'\1• ', answer)

    # Drop standalone bullet artifacts at the very start
    answer = re.sub(r'^[\s•\-\*]+', '', answer).strip()

    if not answer:
        return answer

    # Ensure answer starts with a capital letter (only if first char is a letter)
    if answer[0].isalpha() and answer[0].islower():
        answer = answer[0].upper() + answer[1:]

    # Ensure answer ends with proper punctuation — but don't break a bulleted list
    stripped = answer.rstrip()
    if stripped and stripped[-1] not in '.!?':
        # If the answer is a bulleted list, leave it alone
        if '•' not in stripped.split('\n')[-1]:
            answer = stripped + '.'

    return answer.strip()


def generate_answer(query_text, sections):
    """Generate an accurate answer using NLP techniques on retrieved content.

    The process:
    1. The embedding model retrieves the most relevant document sections
       (semantic similarity — already done before this function is called)
    2. The NLP model analyzes the query and extracts the precise answer
       from the retrieved content
    3. Both the extracted answer and the source section are returned

    This ensures the answer is:
    - Accurate (NLP model understands what is being asked)
    - Strictly based on retrieved content (can only answer from given text)
    - Not unsupported or irrelevant (extracted, not generated freely)

    Returns a dict with 'answer' (NLP extracted) and 'context' (full section).
    """
    if not sections:
        return {'answer': NOT_FOUND_MSG, 'context': ''}

    # Build context from top retrieved sections (cleaned)
    context_parts = []
    total_words = 0
    for section in sections[:5]:
        cleaned = _clean_text(section['section_text'])
        words = len(cleaned.split())
        if total_words + words > 1500:
            break
        context_parts.append(cleaned)
        total_words += words

    context = ' '.join(context_parts)
    full_section = _clean_text(sections[0]['section_text'])

    if not context or len(context) < 20:
        return {'answer': NOT_FOUND_MSG, 'context': ''}

    # Use NLP techniques to analyze the query and extract the answer
    try:
        answer, confidence = _extract_answer(query_text, context)

        if len(answer) > 10:
            return {'answer': _polish_answer(answer), 'context': full_section}
        else:
            best = _pick_best_section(query_text, sections, context)
            return {'answer': _polish_answer(best), 'context': ''}

    except Exception:
        best = _pick_best_section(query_text, sections, context)
        return {'answer': _polish_answer(best), 'context': ''}
