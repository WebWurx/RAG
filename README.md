# Smart Doc Query
### Retrieval-Augmented AI System for Intelligent Document Querying

A local, web-based application where users upload PDF and TXT documents and ask natural language questions about their content. The system uses semantic vector embeddings and cosine similarity to find the most relevant sections and extract an answer — running fully offline after initial setup.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Module Breakdown](#module-breakdown)
5. [Database Design](#database-design)
6. [System Flow](#system-flow)
   - [Document Upload Flow](#document-upload-flow)
   - [Query & Answer Flow](#query--answer-flow)
7. [Routes Reference](#routes-reference)
8. [Frontend Pages](#frontend-pages)
9. [Configuration Parameters](#configuration-parameters)
10. [How to Run](#how-to-run)

---

## Overview

Smart Doc Query is a RAG (Retrieval-Augmented Generation) system built with Python and Flask:

1. Documents are split into overlapping text chunks
2. Each chunk is encoded into a 384-dimension vector using a sentence transformer model
3. When a user asks a question, the question is embedded into the same vector space
4. Cosine similarity ranks which chunks are most semantically related to the question
5. An extractive answer is assembled from the top-scoring sentences in those chunks
6. The answer is displayed with source citations showing document name and page numbers

No external APIs. Everything runs locally.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| Auth | Flask sessions, Werkzeug password hashing (PBKDF2) |
| Database | SQLite (via Python `sqlite3`) |
| PDF Extraction | PyPDF2 |
| Embeddings | `sentence-transformers` — model `all-MiniLM-L6-v2` (~80 MB, downloaded once) |
| Similarity Search | `scikit-learn` — `cosine_similarity` |
| Vector Storage | Python `pickle` blobs stored in SQLite BLOB column |
| Frontend | Bootstrap 5 (SB Admin Pro theme), Jinja2 templates |
| File Serving | Flask `send_from_directory` |

---

## Project Structure

```
RAG-hiba/
├── README.md
├── docs/
│   ├── DESIGN PHASE - HIBA.pdf     Design specification document
│   └── STUDY20-20DOCUMENT.pdf      Reference study document
└── smart_doc_query/
    ├── app.py                      Flask app — all routes and request handling
    ├── config.py                   App configuration (secret key, paths, allowed extensions)
    ├── database.py                 SQLite connection, schema init, query/insert helpers
    ├── document_processor.py       PDF/TXT text extraction and chunking
    ├── retrieval.py                Embeddings, similarity search, answer generation
    ├── requirements.txt            Python dependencies
    ├── run.bat                     Windows one-click launcher
    ├── smart_doc.db                SQLite database (auto-created on first run)
    ├── uploads/                    Uploaded files (auto-created on first run)
    ├── static/
    │   └── style.css               Custom CSS (SB Admin Pro theme variables)
    └── templates/
        ├── base.html               Base layout — navbar, flash messages, shared styles
        ├── register.html           Registration form
        ├── login.html              Login form
        ├── dashboard.html          Upload form + document list + preview modal
        ├── query.html              Question input form
        └── answer.html             Answer display with source citations
```

---

## Module Breakdown

### `config.py`
App-level constants:
- `SECRET_KEY` — Flask session signing key
- `UPLOAD_FOLDER` — absolute path to the `uploads/` directory
- `DATABASE` — absolute path to `smart_doc.db`
- `ALLOWED_EXTENSIONS` — `{'pdf', 'txt'}`

### `database.py`
Three functions handle all database access:
- `init_db()` — creates all 6 tables on startup if they don't exist; includes a migration to add `page_number` to older databases
- `query_db(sql, args, one)` — SELECT helper; returns a single row or list of rows as `sqlite3.Row` objects (dict-accessible)
- `insert_db(sql, args)` — INSERT/DELETE/UPDATE helper; returns `lastrowid`

### `document_processor.py`
Text extraction and chunking:
- `extract_text_from_pdf(filepath)` — uses PyPDF2; returns `[(page_number, text), ...]` per page
- `extract_text_from_txt(filepath)` — reads plain text; returns a single string
- `chunk_pdf_pages(pages, chunk_size=175, overlap=25)` — splits per-page text into word-based chunks with overlap; returns `[(chunk_text, page_number), ...]`
- `chunk_text(text, chunk_size=175, overlap=25)` — same logic for TXT files; all chunks get `page_number=1`
- `_clean_text(text)` — normalises encoding artifacts (replaces `□` and `\ufffd`)

Chunking uses a sliding window: step = `chunk_size - overlap` = 150 words, so consecutive chunks share 25 words, preserving context at chunk boundaries.

### `retrieval.py`
The core NLP module. The sentence transformer model loads once at import time and is cached after first download.

**Query expansion** — a lookup table maps short queries to longer equivalents before embedding:
- `"what is this project"` → `"what is the purpose description and overview of this project system"`
- `"who made this"` → `"who is the author submitted by name of this project"`
- and several similar mappings

**`detect_question_type(query_text)`** — classifies the query as `'definition'`, `'list'`, or `'general'` using grammar structure only (no topic-specific knowledge):
- `definition` — "what is X", "define X", "what does X mean", "what is meant by X"
- `list` — contains a count word or digit + a list noun ("two aspects", "three types"), or starts with "list the" / "name the"
- `general` — everything else

**`extract_list_count(query_text)`** — parses the requested count from a list question ("two" → 2, "3" → 3). Returns `None` if no count is found.

**`extract_list_items(section_text)`** — finds list-structured content in any document section using text formatting patterns: numbered lines (`1.`, `(1)`, `a)`), bullet markers (`-`, `•`, `*`), and ordinal sentence starters (First, Second, Third…).

**`embed_to_blob(text)`** — encodes text to a 384-float numpy vector and serialises it with `pickle` for SQLite BLOB storage.

**`get_relevant_sections(query_text, top_k=8)`**:
1. Expands the query via the lookup table
2. Encodes the expanded query to a vector
3. Loads all `DOCUMENT_SECTION` embeddings from the database
4. Computes cosine similarity between the query vector and each section vector
5. Deduplicates chunks sharing the same 80-character prefix
6. Drops chunks with similarity below 0.25
7. Returns the top `top_k` results sorted by score

**`generate_answer(query_text, sections)`** — branches by question type:

| Type | Logic | Max output | Score threshold |
|------|-------|-----------|----------------|
| `list` | Extract list items from sections → score with cosine similarity → return top N | N from query | — |
| `definition` | Sentence cosine scoring, no position bonus | 2 sentences | 50% of top score |
| `general` | Sentence cosine scoring with position bonus (+0.05 for top section) | 6 sentences | 25% of top score |

Each path falls through to `general` if not enough content is found.

### `app.py`
Flask application with 10 routes. A `@login_required` decorator protects all authenticated routes. On startup it calls `database.init_db()` and creates the uploads directory.

---

## Database Design

Six tables:

```
USER
├── user_id      INTEGER PK AUTOINCREMENT
├── username     TEXT UNIQUE NOT NULL
├── email        TEXT UNIQUE NOT NULL
└── password     TEXT NOT NULL  (PBKDF2 hash via Werkzeug)

DOCUMENT
├── document_id   INTEGER PK AUTOINCREMENT
├── user_id       INTEGER FK → USER
├── document_name TEXT NOT NULL
├── upload_date   TEXT NOT NULL  (YYYY-MM-DD HH:MM)
└── file_type     TEXT NOT NULL  ('pdf' or 'txt')

DOCUMENT_SECTION
├── section_id    INTEGER PK AUTOINCREMENT
├── document_id   INTEGER FK → DOCUMENT
├── section_text  TEXT NOT NULL
├── page_number   INTEGER NOT NULL DEFAULT 1
└── embedding     BLOB  (pickled numpy float32 array, 384 dimensions)

QUERY
├── query_id    INTEGER PK AUTOINCREMENT
├── user_id     INTEGER FK → USER
├── query_text  TEXT NOT NULL
└── query_date  TEXT NOT NULL

RETRIEVAL_RESULT
├── result_id        INTEGER PK AUTOINCREMENT
├── query_id         INTEGER FK → QUERY
├── section_id       INTEGER FK → DOCUMENT_SECTION
└── similarity_score REAL NOT NULL

ANSWER
├── answer_id   INTEGER PK AUTOINCREMENT
├── query_id    INTEGER FK → QUERY
└── answer_text TEXT NOT NULL
```

Relationships:
- One USER → many DOCUMENTs
- One DOCUMENT → many DOCUMENT_SECTIONs
- One USER → many QUERYs
- One QUERY → many RETRIEVAL_RESULTs (one per matched section)
- One QUERY → one ANSWER

---

## System Flow

### Document Upload Flow

```
User selects a PDF or TXT file on the Dashboard
        ↓
POST /upload
        ↓
File extension validated → saved to uploads/
        ↓
DOCUMENT record inserted into DB (returns doc_id)
        ↓
        ├── PDF → PyPDF2 extracts text per page → [(page_num, text), ...]
        └── TXT → read file → single string
        ↓
Chunking: sliding window (175 words, 25-word overlap)
→ [(chunk_text, page_number), ...]
        ↓
For each chunk:
    sentence-transformers encodes chunk → 384-float vector
    vector serialised with pickle → BLOB
    DOCUMENT_SECTION row inserted (text + page_num + embedding BLOB)
        ↓
Redirect to Dashboard
```

### Query & Answer Flow

```
User types a question on the Query page
        ↓
POST /query
        ↓
QUERY record inserted into DB (returns query_id)
        ↓
retrieval.get_relevant_sections(query_text, top_k=8)
    ├── Query expansion (lookup table)
    ├── Encode expanded query → 384-float vector
    ├── Load ALL DOCUMENT_SECTION embeddings from DB
    ├── Cosine similarity: query_vec vs each section_vec
    ├── Deduplicate near-identical chunks (first 80 chars)
    ├── Drop sections with similarity_score < 0.25
    └── Return top 8 by score
        ↓
For each relevant section:
    RETRIEVAL_RESULT row inserted (query_id, section_id, score)
        ↓
retrieval.generate_answer(query_text, sections)
    ├── detect_question_type() → 'definition' | 'list' | 'general'
    │
    ├── LIST path:
    │   ├── extract_list_count() → N
    │   ├── extract_list_items() on each section (numbered/bulleted/ordinal lines)
    │   ├── Score items via cosine similarity
    │   └── Return top N items joined by \n\n
    │
    ├── DEFINITION path:
    │   ├── Split sections into sentences (min 20 chars)
    │   ├── Encode + cosine score (no position bonus)
    │   ├── Drop sentences below 50% of top score
    │   └── Return up to 2 sentences
    │
    └── GENERAL path:
        ├── Split sections into sentences (min 20 chars)
        ├── Encode + cosine score + position bonus (+0.05, top section)
        ├── Drop sentences below 25% of top score
        └── Return up to 6 sentences joined by \n\n
        ↓
ANSWER record inserted
        ↓
Redirect to /answer/<query_id>
        ↓
Answer page:
    - Displays generated answer text
    - Joins RETRIEVAL_RESULT → DOCUMENT_SECTION → DOCUMENT
    - Builds source map: {document_name: [sorted page numbers]}
    - Displays citations below the answer
```

---

## Routes Reference

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET | `/` | No | Redirects to `/login` |
| GET/POST | `/register` | No | User registration |
| GET/POST | `/login` | No | User login |
| GET | `/logout` | Yes | Clears session, redirects to login |
| GET | `/dashboard` | Yes | Upload form + user's document list |
| POST | `/upload` | Yes | File upload, text extraction, embedding, DB insert |
| POST | `/delete/<document_id>` | Yes | Deletes document record, sections, retrieval results, and physical file |
| GET | `/document/<document_id>/file` | Yes | Serves the raw file (used by the inline preview modal) |
| GET/POST | `/query` | Yes | Question input; POST runs retrieval and answer generation |
| GET | `/answer/<query_id>` | Yes | Displays saved answer with source citations |

---

## Frontend Pages

| Template | Purpose | Notable Features |
|----------|---------|-----------------|
| `base.html` | Shared layout | SB Admin Pro navbar, Bootstrap 5, flash message block |
| `register.html` | Registration form | Username, email, password |
| `login.html` | Login form | Username + password |
| `dashboard.html` | Main workspace | Upload form, document table with file type badge and upload date, **inline PDF/TXT preview modal** (full-screen iframe overlay), delete with confirmation dialog |
| `query.html` | Question input | Single textarea, submit button |
| `answer.html` | Answer display | Answer text block + citations table (document name + sorted page numbers) |

The **document preview modal** opens the raw file inside a full-screen iframe. Clicking outside the modal or the × button closes it and clears the iframe `src`.

---

## Configuration Parameters

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `CHUNK_SIZE` | 175 words | `document_processor.py` | Words per document section |
| `OVERLAP` | 25 words | `document_processor.py` | Shared words between consecutive chunks |
| `TOP_K` | 8 | `app.py` | Max sections retrieved per query |
| `RELEVANCE_THRESHOLD` | 0.25 | `retrieval.py` | Minimum cosine similarity to include a section |
| Answer score filter (general) | 25% of top | `retrieval.py` | Minimum sentence score for general questions |
| Answer score filter (definition) | 50% of top | `retrieval.py` | Tighter threshold for definition questions |
| Max answer sentences (general) | 6 | `retrieval.py` | Hard cap for general questions |
| Max answer sentences (definition) | 2 | `retrieval.py` | Hard cap for definition questions |
| Max list items | N from query | `retrieval.py` | Returns exactly the count requested (e.g. "two aspects" → 2) |
| Position bonus | +0.05 | `retrieval.py` | Score boost for first 3 sentences of top section (general path only) |
| Embedding model | `all-MiniLM-L6-v2` | `retrieval.py` | 384-dim output, ~80 MB, downloaded once and cached |

---

## How to Run

### Prerequisites
- Python 3.8+
- Internet connection on **first run only** (to download the `all-MiniLM-L6-v2` model, ~80 MB)

### Windows (easiest)
1. Clone or unzip the repo
2. Double-click `smart_doc_query/run.bat`
3. Open `http://127.0.0.1:5000` in your browser

### Any OS (command line)
```bash
cd smart_doc_query
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5001`.

### First run
On first startup the app automatically:
- Creates `smart_doc.db` with all 6 tables
- Creates the `uploads/` directory
- Downloads and caches the sentence transformer model (~80 MB)
