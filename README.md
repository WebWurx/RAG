# Smart Doc Query
### Retrieval-Augmented AI System for Intelligent Document Querying

A local, web-based application where users upload PDF and TXT documents and ask natural language questions about their content. The system uses semantic vector embeddings and cosine similarity to find the most relevant sections, then applies an NLP extractive question-answering model to produce a precise answer — running fully offline after initial setup.

No external APIs. Everything runs locally.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [How It Works — Technical Deep-Dive](#how-it-works--technical-deep-dive)
4. [Project Structure](#project-structure)
5. [Module Breakdown](#module-breakdown)
6. [Database Design](#database-design)
7. [System Flow](#system-flow)
   - [Document Upload Flow](#document-upload-flow)
   - [Query & Answer Flow](#query--answer-flow)
8. [Routes Reference](#routes-reference)
9. [Frontend Pages](#frontend-pages)
10. [Configuration Parameters](#configuration-parameters)
11. [Dependencies — Packages & Why They're Used](#dependencies--packages--why-theyre-used)
12. [How to Run](#how-to-run)
13. [Document Terms Checklist](#document-terms-checklist)

---

## Overview

Smart Doc Query is a RAG (Retrieval-Augmented Generation) system built with Python and Flask:

1. Documents are split into overlapping, sentence-aware text chunks
2. Each chunk is encoded into a 384-dimension vector using the `all-MiniLM-L6-v2` sentence transformer model
3. Vectors are serialized with `pickle` and stored as BLOBs in SQLite — computed once at upload time
4. When a user asks a question, the question is embedded into the same vector space
5. Cosine similarity ranks which chunks are most semantically related to the question
6. The `distilbert-base-cased-distilled-squad` NLP model extracts the precise answer span from the top-scoring sections
7. If NLP extraction produces a short answer, a heuristic fallback selects the best-matching section
8. The answer is displayed alongside the source context section

---

## Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| Backend | Python 3, Flask | Web framework with Jinja2 templating |
| Auth | Flask sessions | Session-based login (`session['user_id']`, `session['username']`) |
| Database | SQLite via Python `sqlite3` | Lightweight embedded DB, `sqlite3.Row` factory for dict-like access |
| PDF Extraction | PyPDF2 | `PdfReader` extracts text from all pages |
| Embeddings | `sentence-transformers` | Model: `all-MiniLM-L6-v2` — 384-dimensional dense vectors, ~80 MB |
| Semantic Search | `scikit-learn` | `cosine_similarity` ranks document sections by relevance to the query |
| NLP / Answer Extraction | `transformers` + `torch` (PyTorch) | Model: `distilbert-base-cased-distilled-squad` — extractive QA, ~250 MB |
| Vector Storage | Python `pickle` | Numpy float32 arrays serialized to SQLite BLOB columns |
| Frontend | Bootstrap 5 (SB Admin Pro theme), Jinja2 | Custom CSS variables, responsive layout |

---

## How It Works — Technical Deep-Dive

### Semantic Embeddings

The `all-MiniLM-L6-v2` model (from the `sentence-transformers` library) maps any text string into a 384-dimensional dense vector. These vectors capture **semantic meaning** — text about similar concepts produces vectors with high cosine similarity, even when completely different words are used.

For example, a query like *"what is the purpose of the project"* will match a document section containing *"the objective of the system is to..."* because both carry similar meaning in vector space. This is fundamentally different from keyword search, which would fail because "purpose" and "objective" are different strings.

Each document chunk is embedded **once at upload time** and the resulting vector is serialized with `pickle` and stored as a BLOB in the `DOCUMENT_SECTION` table. At query time, only the question needs to be embedded — then cosine similarity is computed against the pre-stored vectors. This makes retrieval fast even with many sections.

### Cosine Similarity

Cosine similarity measures the angle between two vectors:
- **1.0** = identical meaning (vectors point the same direction)
- **0.0** = completely unrelated (vectors are perpendicular)

The system computes cosine similarity between the query vector and every stored section vector using `scikit-learn`'s `cosine_similarity()`. Sections scoring below **0.25** are discarded as irrelevant. The top **5** results are returned, sorted by score.

### Deduplication & Filtering

Before returning results, the system:
1. **Deduplicates** near-identical chunks by comparing their first 100 characters (normalized to lowercase, with leading page numbers stripped)
2. **Filters** out any section with a cosine similarity score below 0.25

### Extractive Question Answering (NLP)

The `distilbert-base-cased-distilled-squad` model is a DistilBERT transformer fine-tuned on the SQuAD 2.0 (Stanford Question Answering Dataset). Given a question and a context passage, it identifies the **exact span of text** that answers the question.

How it works internally:
1. The question and context are tokenized together as a single sequence (truncated to 512 tokens max)
2. The model produces **start logits** and **end logits** — a confidence score for each token position being the start or end of the answer
3. The tokens between the highest-scoring start and end positions are decoded back to text
4. This is **extractive** QA — the answer is always a direct quote from the document, never hallucinated or freely generated

### Sentence-Aware Chunking

Documents are not split blindly by word count. Instead:
1. Text is first split into **sentences** using `.`, `?`, `!` delimiters (sentences shorter than 10 characters are discarded)
2. Sentences are accumulated into chunks targeting **150 words** per chunk
3. When a chunk exceeds the target, it is saved and a new chunk starts with a **50-word overlap** from the previous chunk
4. The overlap ensures that no context is lost at chunk boundaries — important information near the edge of one chunk is also present in the next

### Heuristic Fallback (`_pick_best_section`)

When the NLP model produces a very short answer (fewer than 10 characters), the system falls back to a heuristic approach:
1. Prefers sections containing **numbered lists** (`1.` pattern) — signals a structured, enumerated answer
2. Prefers sections with the **most keyword matches** from the query (words with 4+ characters)
3. Falls back to the full concatenated context if no section stands out

---

## Project Structure

```
RAG/
├── README.md
├── docs/
│   ├── DESIGN PHASE - HIBA.pdf     Design specification document
│   └── STUDY20-20DOCUMENT.pdf      Reference study document
└── smart_doc_query/
    ├── app.py                      Flask app — all routes and request handling
    ├── config.py                   App configuration (secret key, paths, allowed extensions)
    ├── database.py                 SQLite connection, schema init, query/insert helpers
    ├── document_processor.py       PDF/TXT text extraction and sentence-aware chunking
    ├── retrieval.py                Embeddings, semantic search, NLP answer extraction
    ├── requirements.txt            Python dependencies (7 packages)
    ├── run.bat                     Windows one-click launcher
    ├── smart_doc.db                SQLite database (auto-created on first run)
    ├── uploads/                    Uploaded files (auto-created on first run)
    ├── static/
    │   └── style.css               Custom CSS (SB Admin Pro theme variables)
    └── templates/
        ├── base.html               Base layout — navbar, flash messages, shared styles
        ├── register.html           Registration form
        ├── login.html              Login form
        ├── dashboard.html          Upload form + document list
        ├── query.html              Question input form
        └── answer.html             Answer display with source context
```

---

## Module Breakdown

### `config.py`
App-level constants:
- `SECRET_KEY` — Flask session signing key
- `BASE_DIR` — absolute path to the `smart_doc_query/` directory
- `UPLOAD_FOLDER` — absolute path to the `uploads/` directory
- `DATABASE` — absolute path to `smart_doc.db`
- `ALLOWED_EXTENSIONS` — `{'pdf', 'txt'}`

### `database.py`
Three functions handle all database access:
- `get_db()` — creates SQLite connection with `sqlite3.Row` factory for dict-like access
- `init_db()` — creates all 6 tables on startup if they don't exist; includes a migration to add the `embedding BLOB` column to older databases
- `query_db(sql, args, one)` — SELECT helper; returns a single row or list of rows as `sqlite3.Row` objects
- `insert_db(sql, args)` — INSERT/DELETE/UPDATE helper; returns `lastrowid`

### `document_processor.py`
Text extraction and sentence-aware chunking:
- `_clean_text(text)` — normalizes PDF artifacts: removes replacement characters (`\ufffd`), normalizes bullet characters to `•`, removes standalone page numbers, fixes multiple spaces, fixes common PDF word breaks (e.g., `soft ware` → `software`)
- `extract_text_from_pdf(filepath)` — uses PyPDF2 `PdfReader`; reads all pages, concatenates text, returns a single cleaned string
- `extract_text_from_txt(filepath)` — reads plain text with UTF-8 encoding (error-tolerant); returns a single cleaned string
- `_split_sentences(text)` — splits text on `.`, `?`, `!` followed by whitespace; filters out sentences shorter than 10 characters
- `chunk_text(text)` — sentence-aware chunking: accumulates sentences into chunks of ~150 words, with 50-word overlap between consecutive chunks; returns a list of chunk strings

### `retrieval.py`
The core NLP module. Two models are loaded at import time and cached after first download.

**Models:**
1. `SentenceTransformer('all-MiniLM-L6-v2')` — 384-dimensional sentence embeddings for semantic similarity (~80 MB)
2. `distilbert-base-cased-distilled-squad` via `AutoTokenizer` + `AutoModelForQuestionAnswering` — extractive question answering (~250 MB)

**Functions:**
- `embed_to_blob(text)` — encodes text to a 384-float vector via the sentence transformer; serializes with `pickle` for BLOB storage
- `blob_to_array(blob)` — deserializes a pickled vector back to a numpy array
- `get_relevant_sections(query_text, top_k=5)` — semantic similarity retrieval:
  1. Encodes the query to a vector
  2. Loads all `DOCUMENT_SECTION` embeddings from the database (or encodes on-the-fly if the embedding is missing)
  3. Computes cosine similarity for each section
  4. Deduplicates near-identical chunks (first 100 chars, normalized)
  5. Filters out sections with similarity below 0.25
  6. Returns the top `top_k` results sorted by score
- `_extract_answer(question, context)` — uses the distilbert QA model:
  1. Tokenizes question + context together (truncated to 512 tokens)
  2. Computes start/end logits for the answer span
  3. Decodes the best span back to text
  4. Returns `(answer_text, confidence_score)`
- `_clean_text(text)` — display cleanup: removes leading page numbers, normalizes bullets, fixes PDF word breaks
- `_pick_best_section(query_text, sections, fallback_context)` — heuristic fallback:
  1. Scores each of the top 3 sections: +10 for containing a numbered list, +1 per query keyword match
  2. Returns the highest-scoring section, or falls back to the concatenated context
- `generate_answer(query_text, sections)` — main answer generation pipeline:
  1. If no sections, returns the "not found" message
  2. Builds a context string from the top 3 sections (capped at 800 words total)
  3. Calls `_extract_answer()` with the question and context
  4. If the extracted answer is long enough (>10 chars), returns it with the full source section as context
  5. If the answer is too short or extraction fails, falls back to `_pick_best_section()`
  6. Returns a dict: `{'answer': str, 'context': str}`

### `app.py`
Flask application with 8 routes. A `@login_required` decorator protects all authenticated routes. On startup it calls `database.init_db()` and creates the uploads directory.

---

## Database Design

Six tables as specified in the Design Phase document (pages 9-10):

```
USER
├── user_id      INTEGER PK AUTOINCREMENT
├── username     VARCHAR(100) UNIQUE NOT NULL
├── email        VARCHAR(100) UNIQUE NOT NULL
└── password     VARCHAR(100) NOT NULL

DOCUMENT
├── document_id   INTEGER PK AUTOINCREMENT
├── user_id       INTEGER FK → USER
├── document_name VARCHAR(150) NOT NULL
├── upload_date   DATE NOT NULL
└── file_type     VARCHAR(20) NOT NULL  ('pdf' or 'txt')

DOCUMENT_SECTION
├── section_id    INTEGER PK AUTOINCREMENT
├── document_id   INTEGER FK → DOCUMENT
├── section_text  TEXT NOT NULL
└── embedding     BLOB  (pickled numpy float32 array, 384 dimensions)

QUERY
├── query_id    INTEGER PK AUTOINCREMENT
├── user_id     INTEGER FK → USER
├── query_text  TEXT NOT NULL
└── query_date  DATE NOT NULL

RETRIEVAL_RESULT
├── result_id        INTEGER PK AUTOINCREMENT
├── query_id         INTEGER FK → QUERY
├── section_id       INTEGER FK → DOCUMENT_SECTION
└── similarity_score FLOAT NOT NULL

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
        ├── PDF → PyPDF2 extracts text from all pages → single string
        └── TXT → read file → single string
        ↓
Text cleaned (_clean_text: fix PDF artifacts, bullets, word breaks)
        ↓
Sentence-aware chunking (150-word chunks, 50-word overlap)
→ [chunk_text, chunk_text, ...]
        ↓
For each chunk:
    sentence-transformers encodes chunk → 384-float vector
    vector serialised with pickle → BLOB
    DOCUMENT_SECTION row inserted (text + embedding BLOB)
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
retrieval.get_relevant_sections(query_text, top_k=5)
    ├── Encode query → 384-float vector (sentence-transformers)
    ├── Load ALL DOCUMENT_SECTION embeddings from DB
    ├── Cosine similarity: query_vec vs each section_vec (scikit-learn)
    ├── Deduplicate near-identical chunks (first 100 chars)
    ├── Drop sections with similarity_score < 0.25
    └── Return top 5 by score
        ↓
For each relevant section:
    RETRIEVAL_RESULT row inserted (query_id, section_id, score)
        ↓
retrieval.generate_answer(query_text, sections)
    ├── Build context from top 3 sections (max 800 words)
    ├── _extract_answer(question, context)
    │   ├── Tokenize question + context (distilbert, max 512 tokens)
    │   ├── Compute start/end logits for answer span
    │   └── Decode best span → answer text + confidence
    │
    ├── If answer > 10 chars → return extracted answer + source section
    └── If answer too short or extraction fails:
        └── _pick_best_section() heuristic fallback
            ├── Prefer sections with numbered lists (+10)
            ├── Prefer sections with query keyword matches (+1 each)
            └── Fall back to concatenated context
        ↓
ANSWER record inserted
        ↓
Redirect to /answer/<query_id>
        ↓
Answer page displays:
    - Generated answer text
    - Source context section (if available)
```

---

## Routes Reference

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET | `/` | No | Redirects to `/login` |
| GET/POST | `/register` | No | User registration — inserts into USER table |
| GET/POST | `/login` | No | User login — validates against USER table, sets session |
| GET | `/logout` | Yes | Clears session, redirects to login |
| GET | `/dashboard` | Yes | Upload form + user's document list from DOCUMENT table |
| POST | `/upload` | Yes | File upload, text extraction, chunking, embedding, DB insert |
| GET/POST | `/query` | Yes | Question input; POST runs retrieval and answer generation |
| GET | `/answer/<query_id>` | Yes | Displays saved answer with source context |

---

## Frontend Pages

| Template | Purpose | Notable Features |
|----------|---------|-----------------|
| `base.html` | Shared layout | SB Admin Pro navbar, Bootstrap 5, flash message block, footer |
| `register.html` | Registration form | Username, email, password fields |
| `login.html` | Login form | Username + password fields |
| `dashboard.html` | Main workspace | Upload form, document table with file type badge and upload date |
| `query.html` | Question input | Single textarea, submit button |
| `answer.html` | Answer display | Answer text block + source context section (if available) |

---

## Configuration Parameters

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `CHUNK_SIZE` | 150 words | `document_processor.py` | Target words per document section |
| `OVERLAP` | 50 words | `document_processor.py` | Shared words between consecutive chunks |
| `TOP_K` | 5 | `retrieval.py` | Max sections retrieved per query |
| `RELEVANCE_THRESHOLD` | 0.25 | `retrieval.py` | Minimum cosine similarity to include a section |
| Max context words | 800 | `retrieval.py` | Word cap when building context for the QA model |
| Min answer length | 10 chars | `retrieval.py` | Below this, NLP extraction falls back to heuristic |
| Max tokens (QA model) | 512 | `retrieval.py` | Tokenizer truncation limit for question + context |
| Embedding model | `all-MiniLM-L6-v2` | `retrieval.py` | 384-dim output, ~80 MB, downloaded once and cached |
| QA model | `distilbert-base-cased-distilled-squad` | `retrieval.py` | Extractive QA, ~250 MB, downloaded once and cached |

---

## Dependencies — Packages & Why They're Used

All dependencies are listed in `requirements.txt`:

| Package | Why It's Used |
|---------|---------------|
| `flask` | Lightweight Python web framework. Handles HTTP routing, Jinja2 HTML templates, session management, and file uploads. Chosen for simplicity and minimal configuration. |
| `PyPDF2` | Pure-Python PDF reader. Extracts text from uploaded PDF files page-by-page using `PdfReader`. Requires no external C libraries or system dependencies. |
| `sentence-transformers` | Provides pre-trained transformer models that encode text into dense vector embeddings. The `all-MiniLM-L6-v2` model maps any text to a 384-dimensional vector that captures semantic meaning. This is the core of the semantic search — it enables finding relevant content even when completely different words are used (e.g., "purpose" matches "objective"). |
| `scikit-learn` | Machine learning library. Used specifically for `cosine_similarity()` to compute how similar two vectors are. Cosine similarity measures the angle between vectors: 1.0 means identical meaning, 0.0 means completely unrelated. This is how the system ranks document sections by relevance to the user's question. |
| `numpy` | Numerical computing library. Handles the 384-dimensional float vectors produced by the embedding model. Vectors are stored as numpy arrays and serialized via `pickle` for database BLOB storage. |
| `transformers` | Hugging Face library providing access to pre-trained NLP models. Used for the `distilbert-base-cased-distilled-squad` extractive QA model — `AutoTokenizer` tokenizes question + context into model-ready input, and `AutoModelForQuestionAnswering` predicts the start/end positions of the answer span within the context. |
| `torch` | PyTorch deep learning framework. Required runtime backend for the `transformers` QA model. Handles tensor operations, model inference (via `torch.no_grad()` for efficient read-only computation), and logit calculation for answer extraction. |

**Model downloads (first run only):**
- `all-MiniLM-L6-v2` — ~80 MB sentence embedding model, cached locally after download
- `distilbert-base-cased-distilled-squad` — ~250 MB QA model + tokenizer, cached locally after download

After the first run, the system works **fully offline** with no internet connection required.

---

## How to Run

### Prerequisites
- Python 3.8+
- Internet connection on **first run only** (to download the two NLP models, ~330 MB total)

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
Then open `http://127.0.0.1:5000`.

### First run
On first startup the app automatically:
- Creates `smart_doc.db` with all 6 tables
- Creates the `uploads/` directory
- Downloads and caches the sentence transformer model (~80 MB)
- Downloads and caches the distilbert QA model (~250 MB)

---

## Document Terms Checklist

Cross-reference of all terms from the Design Phase and Study Phase documents against the actual implementation.

### Design Phase (`DESIGN PHASE - HIBA.pdf`)

| # | Document Term | Status | Where in Code |
|---|--------------|--------|---------------|
| 1 | Input Design | Done | Registration, login, upload, and query input forms (`templates/register.html`, `login.html`, `dashboard.html`, `query.html`) |
| 2 | Output Design | Done | Answer display with source context (`templates/answer.html`) |
| 3 | Dataflow Diagram — Level 0 (Context) | Done | User ↔ Flask App ↔ SQLite Database (`app.py`) |
| 4 | Dataflow Diagram — Level 1 (User Flow) | Done | register → login → upload document → ask query → generate answer (`app.py` routes) |
| 5 | Entity Relationship Diagram | Done | User *uploads* Document *contains* Document_section; Query *retrieves* Document_section; Query *generates* Answer (`database.py`) |
| 6 | Database Design | Done | SQLite with 6 normalized tables (`database.py:init_db()`) |
| 7 | Normalization (1NF, 2NF, 3NF) | Done | All tables have primary keys (1NF), no partial dependencies (2NF), no transitive dependencies (3NF) |
| 8 | Table 1: USER | Done | `user_id`, `username`, `email`, `password` (`database.py:19-24`) |
| 9 | Table 2: DOCUMENT | Done | `document_id`, `user_id`, `document_name`, `upload_date`, `file_type` (`database.py:28-34`) |
| 10 | Table 3: DOCUMENT_SECTION | Done | `section_id`, `document_id`, `section_text` + `embedding BLOB` (`database.py:42-48`) |
| 11 | Table 4: QUERY | Done | `query_id`, `user_id`, `query_text`, `query_date` (`database.py:51-57`) |
| 12 | Table 5: RETRIEVAL_RESULT | Done | `result_id`, `query_id`, `section_id`, `similarity_score` (`database.py:62-68`) |
| 13 | Table 6: ANSWER | Done | `answer_id`, `query_id`, `answer_text` (`database.py:73-78`) |

### Study Phase (`STUDY20-20DOCUMENT.pdf`)

| # | Document Term | Status | Where in Code |
|---|--------------|--------|---------------|
| 1 | Existing System — keyword search limitations | Done | System uses semantic similarity instead of keyword matching (`retrieval.py:get_relevant_sections`) |
| 2 | Proposed System — RAG with NLP | Done | Full RAG pipeline: embed → retrieve → extract answer (`retrieval.py`) |
| 3 | Proposed System — upload PDFs/text files | Done | PDF and TXT upload supported (`app.py:133-188`, `config.py:ALLOWED_EXTENSIONS`) |
| 4 | Proposed System — extract text, divide into sections | Done | PyPDF2 text extraction + sentence-aware chunking (`document_processor.py`) |
| 5 | Proposed System — store with vector representations | Done | Embeddings stored as BLOB in DOCUMENT_SECTION (`app.py:181`, `retrieval.py:embed_to_blob`) |
| 6 | Proposed System — NLP query analysis | Done | distilbert QA model analyzes query + context (`retrieval.py:_extract_answer`) |
| 7 | Proposed System — semantic similarity retrieval | Done | Cosine similarity on sentence-transformer embeddings (`retrieval.py:get_relevant_sections`) |
| 8 | Proposed System — answers grounded in documents | Done | Extractive QA — answers come only from retrieved text, never hallucinated (`retrieval.py:generate_answer`) |
| 9 | Objectives — NLP techniques | Done | `sentence-transformers` for embeddings + `distilbert` for QA (`retrieval.py`) |
| 10 | Objectives — ML-based similarity matching | Done | `cosine_similarity` from `scikit-learn` (`retrieval.py:126`) |
| 11 | Objectives — secure, user-friendly system | Done | Session-based auth + Bootstrap 5 responsive UI (`app.py:login_required`, `templates/`) |
| 12 | Objectives — scalable for multiple documents | Done | All user documents stored, embeddings pre-computed at upload for fast retrieval |
| 13 | Technical Feasibility — Python, Flask, SQLite, NLP libs | Done | Exact stack used as specified (`requirements.txt`) |
| 14 | Economical Feasibility — open-source tools | Done | All 7 packages are free and open-source |
| 15 | Operational Feasibility — simple UI, natural language | Done | Bootstrap 5 web interface, plain English question input |
