# Smart Doc Query — Improvement Log

**Project:** Retrieval-Augmented AI System for Intelligent Document Querying
**Author:** Hiba (CUAYMCA007)
**Date:** April 2026

---

## Overview

This document records three improvements made to the retrieval and answer
generation pipeline. All changes stay strictly within what the Study Phase
and Design Phase documents describe — no new features, no new UI elements,
no new database tables. Only the internal quality of chunking, retrieval,
and answer generation has been improved.

---

## Improvement 1 — Sentence-aware chunking with overlap

**Doc reference:** Study Phase, Page 4 — *"divided into smaller meaningful sections"*

### Before

```
Method:   Hard cut every 150 words
Overlap:  None
```

Text was split at exactly every 150th word regardless of sentence boundaries.
This caused sentences to be cut in half across two chunks:

```
CHUNK 3 (ends mid-sentence):
"...the system retrieves the most relevant document sections based on"

CHUNK 4 (starts mid-sentence):
"semantic similarity. This ensures that related information is identified..."
```

When a query matched the concept of "semantic similarity", neither chunk
alone contained the full thought — retrieval quality suffered.

### After

```
Method:   Sentence-aware boundaries (never splits mid-sentence)
Overlap:  25 words shared between consecutive chunks
```

Chunks now end at sentence boundaries. The last ~25 words of each chunk
are repeated at the start of the next one, so context is never lost:

```
CHUNK 3 (ends at sentence boundary):
"...the system retrieves the most relevant document sections based on
semantic similarity."

CHUNK 4 (starts with overlap from chunk 3):
"...based on semantic similarity. This ensures that related information
is identified even if different words or sentence structures are used."
```

Now a query about "semantic similarity" finds a complete, self-contained
answer in either chunk.

**Files changed:** `document_processor.py`

---

## Improvement 2 — Pre-stored embeddings with filtering

**Doc reference:** Study Phase, Page 4 — *"stored in a database along with
their corresponding vector representations for efficient retrieval"*

### Before

```
On upload:    Store section text only
On query:     Encode ALL sections live → compare → return top 5
Filtering:    None
Dedup:        None
```

Every time a user asked a question, the system re-encoded every single
document section from scratch. With 50 sections this took ~10 seconds.
With 200 sections it became unusable. Low-relevance junk sections were
returned alongside good ones.

### After

```
On upload:    Encode each section → store vector as BLOB in database
On query:     Encode query only → compare against stored vectors
Filtering:    Drop sections scoring below 0.25 similarity
Dedup:        Skip chunks with identical first 80 characters
```

Vectors are computed once at upload time and stored in the database.
Queries now only encode themselves (one operation) and compare against
pre-stored vectors. This is dramatically faster:

```
BEFORE:  Query "What is the objective?" with 50 sections
         → 51 encode operations (1 query + 50 sections)
         → ~10 seconds

AFTER:   Query "What is the objective?" with 50 sections
         → 1 encode operation (query only)
         → 50 cosine similarity comparisons (fast math, no ML)
         → ~0.5 seconds
```

Low-relevance sections (score < 0.25) are filtered out, and near-duplicate
chunks are removed, so the retrieval results are cleaner.

**Files changed:** `retrieval.py`, `database.py` (added `embedding BLOB`
column to DOCUMENT_SECTION), `app.py` (stores embedding at upload time)

**Database change:** Added `embedding BLOB` column to the DOCUMENT_SECTION
table. This directly implements what the Study Phase describes — storing
vector representations alongside the text content.

---

## Improvement 3 — Sentence-level answer generation

**Doc reference:** Study Phase, Page 4 — *"generates an accurate answer
strictly based on the retrieved content"*

### Before

```
Method:   Return the entire top-ranked chunk as-is
Output:   ~150 words of raw text (the full chunk)
```

The answer was just a dump of the highest-scoring section. It contained
the relevant information buried inside irrelevant sentences:

```
QUERY: "What is the objective of this project?"

ANSWER (before):
"The primary objective of this project is to design and develop an
intelligent document querying system using Retrieval-Augmented Artificial
Intelligence that enables users to retrieve accurate answers from uploaded
documents through natural language queries Another important objective is
to overcome the limitations of traditional keyword-based document search
systems by incorporating semantic understanding and contextual information
retrieval The system aims to reduce the time and effort required by users
to manually search through large volumes of documents The project also
aims to apply Natural Language Processing techniques to analyze and process
both document content and user queries effectively By using machine
learning-based similarity matching the system seeks to improve the relevance
and accuracy of retrieved information Additionally the project focuses on
designing a secure and user-friendly system that supports document upload
efficient storage and controlled access The system is intended to be
scalable so that it can handle multiple documents without performance
degradation"
```

One long block. No focus. Includes sentences about scalability and NLP
techniques that don't directly answer "what is the objective".

### After

```
Method:   Split all retrieved sections into sentences
          Score each sentence against the query
          Return top 6 most relevant sentences
Filter:   Drop sentences scoring below 25% of the top score
Dedup:    Skip duplicate sentences
```

The system now extracts individual sentences from all retrieved sections,
scores each one against the query using cosine similarity, and picks only
the most relevant ones:

```
QUERY: "What is the objective of this project?"

ANSWER (after):
The primary objective of this project is to design and develop an
intelligent document querying system using Retrieval-Augmented Artificial
Intelligence that enables users to retrieve accurate answers from uploaded
documents through natural language queries.

The system aims to reduce the time and effort required by users to manually
search through large volumes of documents.

Additionally, the project focuses on designing a secure and user-friendly
system that supports document upload, efficient storage, and controlled
access.
```

Focused. Only the sentences that directly answer the question. Pulled from
across multiple retrieved sections if needed.

**Files changed:** `retrieval.py`

---

## Summary of changes

```
FILE                      WHAT CHANGED
─────────────────────────────────────────────────────────────────
document_processor.py     Sentence-aware chunking + 25-word overlap
retrieval.py              Pre-stored embeddings, filtering, dedup,
                          sentence-level answer generation
database.py               Added embedding BLOB column to
                          DOCUMENT_SECTION + migration
app.py                    Stores embedding at upload time
```

## What did NOT change

```
UNCHANGED                 WHY
─────────────────────────────────────────────────────────────────
Database tables           Still 6 tables, same fields as Design Phase
DFD flow                  Still register → login → upload → query → answer
UI / Templates            No new pages, no new buttons, no new display
Features                  No delete, no preview, no citations added
Tech stack                Still Python + Flask + SQLite
```

All improvements are internal — better implementations of what the Study
and Design documents already describe. The user-facing behaviour is the
same: upload a document, ask a question, get an answer. The answer is
just more accurate now.
