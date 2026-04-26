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
_DEFINITION_PHRASE_PATTERN = re.compile(
    r'\b(stands?\s+for|means?|refers?\s+to|definition\s+of|meaning\s+of)\b',
    re.IGNORECASE
)


def _is_definition_query(query_text):
    if not query_text:
        return False
    if _DEFINITION_QUERY_PATTERN.match(query_text):
        return True
    return bool(_DEFINITION_PHRASE_PATTERN.search(query_text))


_DEFINITION_VERB_PATTERNS = [
    r'stands?\s+for',
    r'refers?\s+to',
    r'can\s+be\s+defined\s+as',
    r'means?',
    r'is\s+a\b|is\s+an\b|is\s+the\b|is\s+defined',
    r'are\s+a\b|are\s+the\b|are\s+defined',
    r'is|are',
]


def _find_definition_sentence(section_text, key_term):
    """Find the strongest definition sentence for key_term in section_text.

    Strategies, in order:
    1. Direct colon-style: "<Term>: <description>" — handles entries like
       "Hooking: changing applicant's execution flow" where the PDF has no
       terminal punctuation, so the sentence-splitter would glue the entry
       onto its neighbors.
    2. Verb-pattern priority over split sentences ("stands for" > "refers to"
       > "means" > "is/are").

    Returns (sentence, priority) or (None, None). Lower priority is stronger.
    """
    if not section_text or not key_term:
        return None, None
    term = re.escape(key_term)

    # Strategy 1: colon-prefixed entry. Boundary is start-of-text, whitespace,
    # or a dash/punctuation so we match "Hooking:" after "– " or sentence end.
    colon_re = re.compile(
        rf'(?:^|(?<=[\s\-–—.!?(]))\b{term}\b\s*:\s*([^\n.!?]{{5,250}})',
        re.IGNORECASE
    )
    m = colon_re.search(section_text)
    if m:
        sentence = f"{key_term}: {m.group(1).strip()}"
        return sentence, 0

    # Strategy 2: verb-pattern priority on split sentences.
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', section_text)
                 if s.strip() and not _is_header_like(s.strip())]
    for priority, verb_pattern in enumerate(_DEFINITION_VERB_PATTERNS, start=1):
        full_pattern = re.compile(
            rf'\b{term}\b[^.!?\n]{{0,80}}?\b(?:{verb_pattern})\b',
            re.IGNORECASE
        )
        for sentence in sentences:
            if full_pattern.search(sentence):
                return sentence, priority
    return None, None


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
    # Strip trailing context like "in rootkits" / "inside X" / "within Y" so
    # "hooking in rootkits" → "hooking" rather than "hooking in rootkits".
    q = re.sub(r'\s+(?:in|inside|within|used\s+in|used\s+by)\s+\w+(?:\s+\w+)?\s*$',
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


def get_relevant_sections(query_text, top_k=8, document_ids=None):
    """Retrieve the most relevant document sections using semantic similarity.

    Study Phase (Page 4): "ensures that related information is identified
    even if different words or sentence structures are used"

    For definition-style queries ("what is X", "what does X stand for",
    "define X"), sections containing definition patterns like "X is",
    "X stands for", "X refers to" receive a small score boost so the
    actual definition ranks above tangential mentions.

    When `document_ids` is a non-empty list, retrieval is scoped to sections
    belonging to those documents only. None / empty means "search all".
    """
    query_vec = embedding_model.encode(query_text).reshape(1, -1)

    if document_ids:
        placeholders = ','.join('?' * len(document_ids))
        rows = database.query_db(
            f'SELECT section_id, document_id, section_text, embedding '
            f'FROM DOCUMENT_SECTION WHERE document_id IN ({placeholders})',
            tuple(document_ids)
        )
    else:
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
    text = re.sub(r'[▯□●▪▸❖❑✓✗\u2022\u2023\u2043\uf0a7\uf0b7\uf0a8\uf0fc\uf0a4\uf0d8]', '• ', text)

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
    r'\b(what\s+are|list|types?\s+of|kinds?\s+of|examples?\s+of|categories\s+of|approaches\s+to|steps\s+(?:in|of|to|taken|that|which|the)|phases\s+of|ways\s+to|services\s+of|benefits?\s+of|symptoms?\s+of|disadvantages\s+of|advantages\s+of|requirements\s+(?:in|of|for))\b',
    re.IGNORECASE
)
# Implicit list query — noun-phrase that ends in a plural collection noun
# (e.g. "secure socket layer protocols" → asking for the protocol list).
_IMPLICIT_LIST_SUFFIX = re.compile(
    r'\b(protocols|services|types|kinds|examples|approaches|phases|benefits|methods|techniques|ways|requirements|symptoms|disadvantages|advantages|features|categories|properties|functionalities|aspects|fields|steps)\s*\??\s*$',
    re.IGNORECASE
)


def _is_list_query(query_text):
    if not query_text:
        return False
    if _LIST_QUERY_PATTERN.search(query_text):
        return True
    return bool(_IMPLICIT_LIST_SUFFIX.search(query_text))


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


_LIST_STOP_WORDS = {
    'what', 'are', 'is', 'the', 'a', 'an', 'of', 'in', 'at', 'on', 'for',
    'with', 'and', 'or', 'how', 'do', 'does', 'to', 'list', 'kinds',
    'examples', 'taken', 'takes', 'take', 'by', 'that', 'which', 'some',
    'few', 'this', 'these', 'those',
}


def _select_relevant_segment(text, query_text):
    """Pick the segment whose preamble best matches `query_text`.

    Anchors are sentence-like fragments ending in ':' (e.g.
    'Following are the steps … at the sender site:'). When a chunk contains
    multiple distinct lists (PGP property bullets + sender-site step bullets),
    this scopes extraction to just the segment the user actually asked about.

    Returns the substring between the best-matching anchor and the next
    anchor, or None if no anchor matches the query well.
    """
    anchor_re = re.compile(
        r'(?:^|(?<=[.!?]\s))([A-Z][^\n.!?]{4,150}?):\s+'
    )
    # Reject preambles that have swallowed a numbered list marker like
    # "Types of Intruders 1) Masquerader" — that's preamble + list-item title,
    # not a real preamble. Without this filter we'd lose the first item.
    anchors = []
    for m in anchor_re.finditer(text):
        preamble = m.group(1)
        if re.search(r'\b\d+[.)]\s', preamble):
            continue
        anchors.append((m.start(), m.end(), preamble))
    if not anchors:
        return None

    query_words = {w for w in re.findall(r'\w+', query_text.lower())
                   if w not in _LIST_STOP_WORDS and len(w) > 2}
    if not query_words:
        return None

    best_score = 0
    best_idx = -1
    for i, (_, _, preamble) in enumerate(anchors):
        preamble_words = {w for w in re.findall(r'\w+', preamble.lower())
                          if w not in _LIST_STOP_WORDS and len(w) > 2}
        overlap = len(query_words & preamble_words)
        if overlap > best_score:
            best_score = overlap
            best_idx = i

    if best_score < 1:
        return None

    seg_start = anchors[best_idx][1]
    seg_end = anchors[best_idx + 1][0] if best_idx + 1 < len(anchors) else len(text)
    return text[seg_start:seg_end]


def _extract_list_items(text, query_text=''):
    """Extract list items from text. Tries strategies in order:

    1. Bullet/number markers (•, -, *, "1)", "1.", PDF "o " artifact).
       When `query_text` is provided, extraction is first scoped to the
       segment whose preamble best matches the query — handles chunks with
       multiple unrelated lists (PGP properties + sender-site steps).
    2. 'Title: description.' inline-list style, anchored at a preamble like
       'Some types of X:' or 'X include:'. Used by sections where the PDF
       lost the bullet markers during extraction (e.g. SSL alert types).
    """
    if not text:
        return []

    # Strategy 1: marker-based extraction (scoped to query-relevant segment)
    items = []
    normalized = re.sub(r'(^|\s)o\s+(?=[A-Z])', r'\1• ', text)
    scoped = normalized
    if query_text:
        segment = _select_relevant_segment(normalized, query_text)
        if segment is not None:
            scoped = segment
    chunks = re.split(r'(?:\n|(?<=\S)\s+(?=•|\d+[.)]\s))', scoped)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r'^(?:•|[-*]|\d+[.)])\s*(.+)$', chunk)
        if m:
            raw = m.group(1).strip()
            # Trim at first sentence boundary so trailing prose from the next
            # section ('... collection. Need for intrusion Monitoring …')
            # doesn't bloat the item past the length cap.
            cut = re.match(r'^(.{5,250}?[.!?])(?:\s+[A-Z]|\s*$)', raw)
            item = (cut.group(1) if cut else raw).strip().rstrip('.,;:')
            if 2 < len(item) < 250:
                items.append(item)
    if len(items) >= 2:
        return items

    # Strategy 2: inline 'Title: description.' items after a preamble.
    # Anchor at the LAST preamble in the chunk so we capture the items right
    # after the relevant heading rather than every "X:" sentence in the chunk.
    anchor_re = re.compile(
        r'\b(?:(?:some\s+)?types?\s+of\s+\w+|kinds?\s+of\s+\w+|examples?\s+of\s+\w+|categories\s+of\s+\w+|are\s+as\s+follows|include[s]?|are)\s*:\s*',
        re.IGNORECASE
    )
    matches = list(anchor_re.finditer(text))
    body = text[matches[-1].end():] if matches else text

    item_re = re.compile(
        r'(?:^|(?<=[.!?]\s))([A-Z][A-Za-z][\w\s\-/]{1,40}):\s+([^.!?]{10,250}?[.!?])'
    )
    inline_items = []
    for m in item_re.finditer(body):
        title = m.group(1).strip()
        desc = m.group(2).strip().rstrip('.!?').rstrip()
        if len(title.split()) > 6:
            continue
        item = f'{title}: {desc}'
        if 8 < len(item) < 280:
            inline_items.append(item)
        if len(inline_items) >= 10:
            break
    if len(inline_items) >= 2:
        return inline_items
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
            items = _extract_list_items(text, query_text)
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

    # List queries: skip the QA model entirely. SQuAD-style models return one
    # narrow span (e.g. just "Handshake failure") and the list extractor in
    # _pick_best_section produces the full bulleted answer. Routed before
    # the definition branch because "what are the types of X" matches both
    # patterns but is fundamentally a list question.
    if _is_list_query(query_text):
        best = _pick_best_section(query_text, sections, context)
        return {'answer': _polish_answer(best), 'context': full_section}

    # Definition queries: prefer an explicit "X stands for / refers to / means / is"
    # sentence. Search top sections with verb-pattern priority — a "PGP stands
    # for …" sentence in chunk #2 beats a "PGP is open source" sentence in chunk #1.
    if _is_definition_query(query_text):
        key_term = _extract_key_term(query_text)
        if key_term:
            best_def = None
            best_priority = None
            for section in sections[:5]:
                cleaned = _clean_text(section['section_text'])
                sentence, priority = _find_definition_sentence(cleaned, key_term)
                if sentence is not None and (best_priority is None or priority < best_priority):
                    best_def = sentence
                    best_priority = priority
                    if priority == 0:
                        break
            if best_def:
                return {'answer': _polish_answer(best_def), 'context': full_section}

    # General queries: use NLP techniques to analyze the query and extract the answer
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
