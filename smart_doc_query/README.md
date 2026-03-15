# Smart Doc Query
### Retrieval-Augmented AI System for Intelligent Document Querying


---

## Overview

Smart Doc Query is an intelligent document querying system that allows users to upload PDF or TXT documents and ask natural language questions about their content. Instead of relying on traditional keyword-based search, the system uses semantic vector embeddings and NLP techniques to understand the meaning behind a query, retrieve the most relevant sections, and generate a precise extractive answer.

---

## Problem Statement

Existing document search tools rely on keyword matching, which:
- Cannot understand the semantic meaning or intent of a query
- Misses relevant information when exact keywords are not present
- Forces users to manually read through multiple documents
- Provides no ranking by relevance or context
- Lacks natural language query support and intelligent answer generation

---

## Proposed Solution

A Retrieval-Augmented AI system that:
- Allows users to upload PDF and TXT files
- Extracts and divides document content into meaningful sections
- Stores sections with vector representations in a database
- Uses NLP to analyze natural language queries
- Retrieves the most relevant sections using semantic similarity
- Generates accurate, context-based answers strictly from retrieved content

---

## Features

- User registration and login (session-based authentication)
- Upload PDF and TXT documents
- Automatic text extraction and chunking with page tracking
- Semantic vector embeddings using `sentence-transformers` (all-MiniLM-L6-v2)
- Cosine similarity search to find relevant sections
- Extractive answer generation from top retrieved sections (up to 6 sentences)
- Answer displayed with inline source citations (document name + page numbers)
- Delete uploaded documents from the dashboard
- Clean Bootstrap 5 interface

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| Database | SQLite |
| PDF Extraction | PyPDF2 |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Similarity Search | scikit-learn (cosine similarity) |
| Frontend | Bootstrap 5, Jinja2 Templates |

All libraries are free and open-source. No external API keys required. Runs fully locally.

---

## Database Design

The system uses 6 tables as per the design specification:

| Table | Key Fields | Purpose |
|-------|-----------|---------|
| USER | user_id (PK), username, email, password | Registration & login |
| DOCUMENT | document_id (PK), user_id (FK), document_name, upload_date, file_type | Track uploaded files |
| DOCUMENT_SECTION | section_id (PK), document_id (FK), section_text, page_number, embedding | Chunked content with vector embeddings |
| QUERY | query_id (PK), user_id (FK), query_text, query_date | Store user questions |
| RETRIEVAL_RESULT | result_id (PK), query_id (FK), section_id (FK), similarity_score | Relevant sections found |
| ANSWER | answer_id (PK), query_id (FK), answer_text | Generated answer |

---

## System Flow

```
Register / Login
      ↓
Upload Document (PDF or TXT)
      ↓
Text Extraction → Chunking → Embedding (all-MiniLM-L6-v2) → Stored in DB
      ↓
User Asks a Question
      ↓
Query Embedded → Cosine Similarity with All Section Vectors
      ↓
Top 5 Relevant Sections Retrieved (similarity threshold ≥ 0.25)
      ↓
Sentences scored and filtered (min 25% of top score) → Up to 6 sentences selected
      ↓
Extractive Answer Generated → Displayed with Source Citations (PDF + page)
```

---

## Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk size | 175 words | Words per document section |
| Chunk overlap | 25 words | Overlap between consecutive chunks |
| Top-K retrieval | 5 sections | Number of sections retrieved per query |
| Relevance threshold | 0.25 | Minimum cosine similarity to include a section |
| Answer score filter | 25% of top | Minimum sentence score relative to best sentence |
| Max answer sentences | 6 | Maximum sentences included in a generated answer |

---

## Project Structure

```
smart_doc_query/
├── app.py                  Main Flask application
├── config.py               Configuration settings
├── database.py             SQLite database setup and helpers
├── document_processor.py   PDF/TXT extraction and chunking
├── retrieval.py            Embeddings, similarity search, answer generation
├── requirements.txt        Python dependencies
├── run.bat                 Windows launcher script
├── smart_doc.db            SQLite database (auto-created on first run)
├── uploads/                Uploaded files (auto-created on first run)
├── static/
│   └── style.css           Custom CSS
└── templates/
    ├── base.html           Base layout with Bootstrap navbar
    ├── register.html       User registration page
    ├── login.html          User login page
    ├── dashboard.html      Upload documents + view uploaded list
    ├── query.html          Ask a question interface
    └── answer.html         Display answer with source citations
```

---

## How to Run (Windows)

### Option 1 — Double-click (easiest)
1. Extract the project zip
2. Double-click `run.bat`
3. Open your browser and go to `http://127.0.0.1:5000`

### Option 2 — Command Prompt
```
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

> **Note:** An internet connection is required on first run to download packages and the NLP model (~80 MB, once only). After that, the system works fully offline.

---

## Feasibility

| Type | Status | Notes |
|------|--------|-------|
| Technical | Feasible | Python, Flask, SQLite, free NLP/ML libraries — minimal hardware requirements |
| Economic | Feasible | All open-source tools, no licensed software or API costs |
| Operational | Feasible | Simple UI, natural language interaction, no advanced technical knowledge needed |

---

## Objectives

- Design an intelligent document querying system using Retrieval-Augmented AI
- Overcome the limitations of keyword-based search using semantic understanding
- Reduce manual document search time through automation
- Apply NLP techniques for effective content analysis
- Improve retrieval accuracy using machine learning similarity matching
- Design a secure, user-friendly system with document upload and storage

---

## Scope

The system is intended for:
- Academic institutions
- Researchers and students
- Office environments with large document collections

It is a local, single-user system designed as a college project. It does not include admin panels, file sharing between users, or cloud deployment.
