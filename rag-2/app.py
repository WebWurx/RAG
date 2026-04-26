"""
Smart Doc Query — Main Flask Application

Retrieval-Augmented AI System for Intelligent Document Querying

System flow as per DFD Level 1 (Design Phase, Page 5):
    User → register → login → upload document → ask query → generate answer

Tech stack as per Study Phase (Page 9):
    Python, Flask, SQLite
"""

import os
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash)
from werkzeug.utils import secure_filename

import config
import database
import document_processor
import retrieval

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
database.init_db()


@app.context_processor
def inject_sidebar_data():
    """Provide sidebar data (documents + recent queries) to all authenticated templates."""
    if 'user_id' in session:
        docs = database.query_db(
            'SELECT * FROM DOCUMENT WHERE user_id = ? ORDER BY upload_date DESC',
            (session['user_id'],)
        )
        recent_queries = database.query_db(
            'SELECT query_id, query_text FROM QUERY WHERE user_id = ? ORDER BY query_id DESC LIMIT 10',
            (session['user_id'],)
        )
        return {'sidebar_docs': docs, 'sidebar_queries': recent_queries}
    return {}


def allowed_file(filename):
    """Check if the file type is PDF or TXT as per Study Phase (Page 4)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def _format_page_label(pages):
    """Compact label for a sorted list of page numbers.
    [3] → "p. 3"   [3,4,5,7] → "pp. 3-5, 7"   [] → ""
    """
    if not pages:
        return ''
    runs = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            runs.append((start, prev))
            start = prev = p
    runs.append((start, prev))
    parts = [str(a) if a == b else f'{a}-{b}' for a, b in runs]
    prefix = 'p.' if len(pages) == 1 else 'pp.'
    return f'{prefix} ' + ', '.join(parts)


def login_required(f):
    """Session-based access control as implied by the login process in DFD."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── Routes (following DFD Level 1) ──────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('query'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration — DFD Level 1, Process 1: register
    Stores data in USER table (Design Phase, Page 9).
    """
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip()
        password = request.form['password'].strip()

        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        existing = database.query_db(
            'SELECT user_id FROM USER WHERE username = ? OR email = ?',
            (username, email), one=True
        )
        if existing:
            flash('Username or email already exists.', 'danger')
            return render_template('register.html')

        database.insert_db(
            'INSERT INTO USER (username, email, password) VALUES (?, ?, ?)',
            (username, email, password)
        )
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login — DFD Level 1, Process 2: login
    Validates against USER table, stores to login data store.
    """
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        user = database.query_db(
            'SELECT * FROM USER WHERE username = ? AND password = ?',
            (username, password), one=True
        )
        if user:
            session['user_id']  = user['user_id']
            session['username'] = user['username']
            session['email']    = user['email']
            return redirect(url_for('query'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    """Upload document — DFD Level 1, Process 3: upload document
    Study Phase (Page 4): "users can upload documents such as PDFs or text files"
    Study Phase (Page 4): "processed to extract textual content, which is then
    divided into smaller meaningful sections"
    Stores to DOCUMENT and DOCUMENT_SECTION tables.
    """
    if 'file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('query'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('query'))

    if not allowed_file(file.filename):
        flash('Only PDF and TXT files are allowed.', 'danger')
        return redirect(url_for('query'))

    filename  = secure_filename(file.filename)
    file_type = filename.rsplit('.', 1)[1].lower()
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Save to DOCUMENT table
    doc_id = database.insert_db(
        'INSERT INTO DOCUMENT (user_id, document_name, upload_date, file_type) VALUES (?, ?, ?, ?)',
        (session['user_id'], filename, datetime.now().strftime('%Y-%m-%d'), file_type)
    )

    # Extract text from uploaded document.
    # PDFs are extracted per-page so each chunk can be tagged with its page.
    # TXT has no pages — page_number stays NULL.
    if file_type == 'pdf':
        pages = document_processor.extract_pages_from_pdf(filepath)
        page_chunks = document_processor.chunk_pages(pages)
    else:
        txt = document_processor.extract_text_from_txt(filepath)
        page_chunks = [(None, c) for c in document_processor.chunk_text(txt)]

    if not page_chunks:
        flash('Could not extract text from the file.', 'danger')
        return redirect(url_for('query'))

    # Store sections with vector representations and page references.
    # Study Phase (Page 4): "stored in a database along with their
    # corresponding vector representations for efficient retrieval"
    for page_num, chunk in page_chunks:
        embedding_blob = retrieval.embed_to_blob(chunk)
        database.insert_db(
            'INSERT INTO DOCUMENT_SECTION (document_id, section_text, embedding, page_number) VALUES (?, ?, ?, ?)',
            (doc_id, chunk, embedding_blob, page_num)
        )
    chunks = page_chunks

    flash(f'"{filename}" uploaded and processed successfully ({len(chunks)} sections).', 'success')
    # ?just_uploaded=<id> tells the sidebar JS to auto-select this doc on landing.
    return redirect(url_for('query', just_uploaded=doc_id))


@app.route('/document/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_document(document_id):
    """Delete a user's document, its sections, related retrieval results, and the file on disk."""
    doc = database.query_db(
        'SELECT * FROM DOCUMENT WHERE document_id = ? AND user_id = ?',
        (document_id, session['user_id']), one=True
    )
    if not doc:
        flash('Document not found.', 'danger')
        return redirect(url_for('query'))

    conn = database.get_db()
    try:
        # Queries that drew on any section of this document
        affected_query_ids = [row['query_id'] for row in conn.execute(
            '''SELECT DISTINCT rr.query_id
               FROM RETRIEVAL_RESULT rr
               JOIN DOCUMENT_SECTION ds ON ds.section_id = rr.section_id
               WHERE ds.document_id = ?''',
            (document_id,)
        ).fetchall()]

        conn.execute(
            '''DELETE FROM RETRIEVAL_RESULT
               WHERE section_id IN (
                   SELECT section_id FROM DOCUMENT_SECTION WHERE document_id = ?
               )''',
            (document_id,)
        )

        # Queries with no remaining retrieval results are now orphans —
        # wipe their answers and the queries themselves.
        for qid in affected_query_ids:
            still_has_results = conn.execute(
                'SELECT 1 FROM RETRIEVAL_RESULT WHERE query_id = ? LIMIT 1', (qid,)
            ).fetchone()
            if not still_has_results:
                conn.execute('DELETE FROM ANSWER WHERE query_id = ?', (qid,))
                conn.execute('DELETE FROM QUERY WHERE query_id = ?', (qid,))

        conn.execute('DELETE FROM DOCUMENT_SECTION WHERE document_id = ?', (document_id,))
        conn.execute(
            'DELETE FROM DOCUMENT WHERE document_id = ? AND user_id = ?',
            (document_id, session['user_id'])
        )
        conn.commit()
    finally:
        conn.close()

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], doc['document_name'])
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    flash(f'"{doc["document_name"]}" deleted.', 'success')
    return redirect(url_for('query'))


@app.route('/query', methods=['GET', 'POST'])
@login_required
def query():
    """Ask query — DFD Level 1, Process 4: ask query
    Study Phase (Page 4): "user enters a question in natural language"
    Stores to QUERY table, retrieves from DOCUMENT_SECTION,
    stores results in RETRIEVAL_RESULT, generates answer in ANSWER.
    """
    if request.method == 'POST':
        query_text = request.form['query_text'].strip()
        if not query_text:
            flash('Please enter a question.', 'danger')
            return render_template('query.html')

        # Optional doc-scope filter from the sidebar selection. The hidden
        # field is a comma-separated list of document_ids; any value not
        # owned by the current user is dropped before reaching retrieval.
        raw_ids = request.form.get('document_ids', '').strip()
        document_ids = None
        if raw_ids:
            requested = [int(x) for x in raw_ids.split(',') if x.strip().isdigit()]
            if requested:
                placeholders = ','.join('?' * len(requested))
                owned = database.query_db(
                    f'SELECT document_id FROM DOCUMENT WHERE user_id = ? AND document_id IN ({placeholders})',
                    (session['user_id'], *requested)
                )
                document_ids = [row['document_id'] for row in owned] or None

        # Save query to QUERY table
        query_id = database.insert_db(
            'INSERT INTO QUERY (user_id, query_text, query_date) VALUES (?, ?, ?)',
            (session['user_id'], query_text, datetime.now().strftime('%Y-%m-%d'))
        )

        # Retrieve relevant sections based on semantic similarity
        sections = retrieval.get_relevant_sections(query_text, document_ids=document_ids)

        # Save retrieval results to RETRIEVAL_RESULT table
        for sec in sections:
            database.insert_db(
                'INSERT INTO RETRIEVAL_RESULT (query_id, section_id, similarity_score) VALUES (?, ?, ?)',
                (query_id, sec['section_id'], sec['score'])
            )

        # Generate answer from retrieved content
        result = retrieval.generate_answer(query_text, sections)
        answer_text = result['answer']
        context_text = result.get('context', '')

        # Save answer to ANSWER table
        database.insert_db(
            'INSERT INTO ANSWER (query_id, answer_text) VALUES (?, ?)',
            (query_id, answer_text)
        )

        # Store context in session for display on answer page
        session['last_context'] = context_text

        return redirect(url_for('answer', query_id=query_id))

    return render_template('query.html')


@app.route('/answer/<int:query_id>')
@login_required
def answer(query_id):
    """Display answer — DFD Level 1, Process 5: generate answer
    Reads from QUERY, ANSWER, and RETRIEVAL_RESULT tables.
    """
    query_row = database.query_db(
        'SELECT * FROM QUERY WHERE query_id = ? AND user_id = ?',
        (query_id, session['user_id']), one=True
    )
    if not query_row:
        flash('Query not found.', 'danger')
        return redirect(url_for('query'))

    answer_row = database.query_db(
        'SELECT * FROM ANSWER WHERE query_id = ?', (query_id,), one=True
    )

    # Get context from session (if available)
    context = session.pop('last_context', '')

    # Fetch recent Q&As for stacked chat history display
    chat_history = database.query_db(
        '''SELECT q.query_id, q.query_text, a.answer_text
           FROM QUERY q LEFT JOIN ANSWER a ON q.query_id = a.query_id
           WHERE q.user_id = ? AND q.query_id <= ?
           ORDER BY q.query_id DESC LIMIT 5''',
        (session['user_id'], query_id)
    )
    chat_history = list(reversed(chat_history))

    # Enrich each chat item with source documents + the page numbers each
    # contributed (e.g. "Answer based on: notes.pdf · pp. 3-5, 7").
    enriched_history = []
    for item in chat_history:
        rows = database.query_db(
            '''SELECT d.document_id, d.document_name, d.file_type, ds.page_number
               FROM RETRIEVAL_RESULT rr
               JOIN DOCUMENT_SECTION ds ON rr.section_id = ds.section_id
               JOIN DOCUMENT d ON ds.document_id = d.document_id
               WHERE rr.query_id = ?''',
            (item['query_id'],)
        )
        # Group by document_id, collect distinct page numbers.
        by_doc = {}
        for r in rows:
            entry = by_doc.setdefault(r['document_id'], {
                'document_name': r['document_name'],
                'file_type': r['file_type'],
                'pages': set(),
            })
            if r['page_number'] is not None:
                entry['pages'].add(r['page_number'])
        source_docs = []
        for entry in by_doc.values():
            pages = sorted(entry['pages'])
            source_docs.append({
                'document_name': entry['document_name'],
                'file_type': entry['file_type'],
                'pages': pages,
                'pages_label': _format_page_label(pages),
            })
        enriched_history.append({
            'query_id': item['query_id'],
            'query_text': item['query_text'],
            'answer_text': item['answer_text'],
            'source_docs': source_docs
        })

    return render_template('answer.html',
                           query=query_row,
                           answer=answer_row,
                           context=context,
                           chat_history=enriched_history)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
