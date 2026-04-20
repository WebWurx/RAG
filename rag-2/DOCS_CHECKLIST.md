# Smart Doc Query — Documentation Checklist

**Project:** Retrieval-Augmented AI System for Intelligent Document Querying
**Author:** Hiba (CUAYMCA007)

This file maps every technical term from the Study Phase and Design Phase
documents to its implementation in the code. Use this for viva preparation.

---

## Study Phase — System Study

### Existing System (Page 3)

| Doc Term | What It Says | Status |
|----------|-------------|--------|
| Keyword-based search | "users depend mainly on traditional keyword-based search methods" | This is what our system replaces — we use semantic similarity instead |
| Exact keyword matching | "require users to enter exact keywords" | Our system uses NLP embeddings so exact keywords are NOT needed |
| No natural language support | "do not support natural language querying" | Our system accepts natural language questions |

### Proposed System (Page 4)

| Doc Term | What It Says | Implementation | File |
|----------|-------------|---------------|------|
| Upload PDF/TXT | "users can upload documents such as PDFs or text files" | Upload route accepts .pdf and .txt | app.py → /upload |
| Text extraction | "processed to extract textual content" | PyPDF2 for PDF, file read for TXT | document_processor.py → extract_text_from_pdf() |
| Divided into sections | "divided into smaller meaningful sections" | Sentence-aware chunking, 150 words, 50 word overlap | document_processor.py → chunk_text() |
| Vector representations | "stored in a database along with their corresponding vector representations" | sentence-transformers (all-MiniLM-L6-v2) encodes text to 384-dim vectors, stored as BLOB | retrieval.py → embed_to_blob() |
| Efficient retrieval | "for efficient retrieval" | Embeddings computed once at upload, only query encoded at search time | retrieval.py → get_relevant_sections() |
| NLP techniques | "analyzes the query using Natural Language Processing techniques" | Transformer-based NLP model (distilbert) analyzes query and extracts answer | retrieval.py → _extract_answer() |
| Semantic similarity | "retrieves the most relevant document sections based on semantic similarity" | Cosine similarity between query embedding and stored section embeddings | retrieval.py → get_relevant_sections() |
| Different words/structures | "related information is identified even if different words or sentence structures are used" | Semantic embeddings capture meaning, not keywords — "firewall types" matches "kinds of firewall" | retrieval.py → embedding_model.encode() |
| Accurate answer | "generates an accurate answer strictly based on the retrieved content" | NLP model extracts answer from retrieved sections; fallback to top chunk | retrieval.py → generate_answer() |
| No unsupported responses | "prevents the system from generating unsupported or irrelevant responses" | Answer is extracted from document content only — model cannot make things up | retrieval.py → _extract_answer() |
| Natural language interaction | "interact with documents in a natural and intelligent manner" | User types questions in plain English, system understands intent | app.py → /query route |

### Objectives (Page 5)

| Doc Term | What It Says | Implementation | File |
|----------|-------------|---------------|------|
| NLP for content and queries | "apply NLP techniques to analyze and process both document content and user queries" | Content: embedded at upload. Queries: embedded + NLP extraction at query time | retrieval.py |
| ML similarity matching | "machine learning-based similarity matching" | ML embedding model + cosine similarity scoring | retrieval.py |
| Secure system | "designing a secure and user-friendly system" | User registration, login required for all actions | app.py → login_required decorator |
| Document upload and storage | "supports document upload, efficient storage" | Upload route + DOCUMENT table + DOCUMENT_SECTION table | app.py, database.py |
| Controlled access | "controlled access" | Session-based auth, users only see their own documents | app.py → session['user_id'] |

### Feasibility — Technical (Page 9)

| Doc Term | What It Says | Implementation | File |
|----------|-------------|---------------|------|
| Python | "implemented using Python" | Entire backend is Python | all .py files |
| Flask | "Flask" | Web framework | app.py |
| SQLite | "SQLite" | Database engine | database.py → sqlite3 |
| NLP and ML libraries | "standard NLP and machine learning libraries" | sentence-transformers, transformers, scikit-learn, numpy | requirements.txt |
| Minimal hardware | "minimal hardware configuration" | Runs on any standard computer, no GPU needed | — |

---

## Design Phase — System Design

### Input Design (Page 2)

| Doc Term | Implementation | File |
|----------|---------------|------|
| User registration input | Username, email, password form | templates/register.html |
| User login input | Username, password form | templates/login.html |
| Document upload input | File upload (PDF/TXT) | templates/dashboard.html |
| Query input | Natural language text area | templates/query.html |

### Output Design (Page 2)

| Doc Term | Implementation | File |
|----------|---------------|------|
| Generated answer display | Answer card with extracted answer | templates/answer.html |
| Document list display | Table showing uploaded documents | templates/dashboard.html |

### DFD Level 1 — System Flow (Page 5)

| Process | Doc Description | Implementation | Route |
|---------|----------------|---------------|-------|
| 1. Register | User → register → login data store | Registration form → USER table | /register |
| 2. Login | User → login → login data store | Auth check → session created | /login |
| 3. Upload document | User → upload → documents + document sections | File saved → text extracted → chunked → embedded → stored | /upload |
| 4. Ask query | User → query → queries + document section | Query saved → sections retrieved by similarity | /query |
| 5. Generate answer | Query → generate answer → answer | NLP extracts answer from retrieved sections → saved | /answer |

### ER Diagram Entities (Page 7)

| Entity | Relationship | Implementation |
|--------|-------------|---------------|
| User | uploads → Document | user_id FK in DOCUMENT table |
| Document | contains → Document_Section | document_id FK in DOCUMENT_SECTION table |
| User | submits → Query | user_id FK in QUERY table |
| Query | retrieves → Document_Section | query_id + section_id in RETRIEVAL_RESULT table |
| Query | generates → Answer | query_id FK in ANSWER table |

### Database Tables (Pages 9-10)

| Table | Doc Fields | Code Fields | Match |
|-------|-----------|-------------|-------|
| USER | user_id, username, email, password | user_id, username, email, password | Yes |
| DOCUMENT | document_id, user_id, document_name, upload_date, file_type | document_id, user_id, document_name, upload_date, file_type | Yes |
| DOCUMENT_SECTION | section_id, document_id, section_text | section_id, document_id, section_text + embedding (BLOB) | Yes + embedding for vector storage |
| QUERY | query_id, user_id, query_text, query_date | query_id, user_id, query_text, query_date | Yes |
| RETRIEVAL_RESULT | result_id, query_id, section_id, similarity_score | result_id, query_id, section_id, similarity_score | Yes |
| ANSWER | answer_id, query_id, answer_text | answer_id, query_id, answer_text | Yes |

### Normalization (Page 8)

| Form | Doc Description | Implementation |
|------|----------------|---------------|
| 1NF | "eliminates duplicate columns, unique primary key" | Each table has INTEGER PRIMARY KEY AUTOINCREMENT |
| 2NF | "removes subset of data, foreign keys" | Foreign keys link tables (user_id, document_id, query_id, section_id) |
| 3NF | "removes columns not dependent on primary key" | No transitive dependencies in any table |

---

## Tech Stack Summary

| Component | Doc Reference | Implementation |
|-----------|--------------|---------------|
| Backend | Study p.9: "Python, Flask" | Flask web app (app.py) |
| Database | Study p.9: "SQLite" | sqlite3 (database.py) |
| PDF extraction | Study p.4: "extract textual content" | PyPDF2 (document_processor.py) |
| Embeddings | Study p.4: "vector representations" | sentence-transformers all-MiniLM-L6-v2 (retrieval.py) |
| Similarity | Study p.5: "ML-based similarity matching" | scikit-learn cosine_similarity (retrieval.py) |
| NLP | Study p.4: "NLP techniques" | transformers distilbert (retrieval.py) |
| Frontend | Study p.10: "simple and intuitive interface" | HTML/CSS templates with clean UI |

---

## Viva Quick Answers

**Q: How does the system retrieve answers?**
A: The system uses semantic similarity. Documents are divided into sections
and each section is converted to a vector using an NLP embedding model.
When a user asks a question, the query is also converted to a vector.
Cosine similarity is computed between the query vector and all section
vectors to find the most relevant sections.

**Q: What NLP techniques are used?**
A: Two NLP models are used. First, sentence-transformers converts text to
semantic vector embeddings for similarity matching. Second, a transformer-based
NLP model analyzes the query and extracts the precise answer from the
retrieved document sections.

**Q: How is the answer generated?**
A: The answer is generated strictly from the retrieved document content.
The NLP model reads the query and the retrieved section, then extracts
the most relevant portion as the answer. It cannot generate information
that is not in the documents.

**Q: What is Retrieval-Augmented AI?**
A: It combines information retrieval (finding relevant documents) with
AI (understanding and extracting answers). The retrieval step finds the
right content, and the AI step generates an accurate answer from it.

**Q: Why not use keyword search?**
A: Keyword search fails when different words are used. For example,
searching "firewall types" would miss a section titled "kinds of firewall."
Semantic similarity understands meaning, so it finds relevant content
regardless of exact wording.

**Q: What database tables are used?**
A: Six tables as per the design document: USER (authentication),
DOCUMENT (uploaded files), DOCUMENT_SECTION (chunked content with
embeddings), QUERY (user questions), RETRIEVAL_RESULT (matched sections
with scores), and ANSWER (generated answers).
