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


def allowed_file(filename):
    """Check if the file type is PDF or TXT as per Study Phase (Page 4)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


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
            return redirect(url_for('dashboard'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard — shows uploaded documents.
    Reads from DOCUMENT table (Design Phase, Page 9).
    """
    docs = database.query_db(
        'SELECT * FROM DOCUMENT WHERE user_id = ? ORDER BY upload_date DESC',
        (session['user_id'],)
    )
    return render_template('dashboard.html', docs=docs)


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
        return redirect(url_for('dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))

    if not allowed_file(file.filename):
        flash('Only PDF and TXT files are allowed.', 'danger')
        return redirect(url_for('dashboard'))

    filename  = secure_filename(file.filename)
    file_type = filename.rsplit('.', 1)[1].lower()
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Save to DOCUMENT table
    doc_id = database.insert_db(
        'INSERT INTO DOCUMENT (user_id, document_name, upload_date, file_type) VALUES (?, ?, ?, ?)',
        (session['user_id'], filename, datetime.now().strftime('%Y-%m-%d'), file_type)
    )

    # Extract text from uploaded document
    if file_type == 'pdf':
        text = document_processor.extract_text_from_pdf(filepath)
    else:
        text = document_processor.extract_text_from_txt(filepath)

    if not text.strip():
        flash('Could not extract text from the file.', 'danger')
        return redirect(url_for('dashboard'))

    # Divide into smaller meaningful sections and store
    chunks = document_processor.chunk_text(text)
    for chunk in chunks:
        database.insert_db(
            'INSERT INTO DOCUMENT_SECTION (document_id, section_text) VALUES (?, ?)',
            (doc_id, chunk)
        )

    flash(f'"{filename}" uploaded and processed successfully ({len(chunks)} sections).', 'success')
    return redirect(url_for('dashboard'))


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

        # Save query to QUERY table
        query_id = database.insert_db(
            'INSERT INTO QUERY (user_id, query_text, query_date) VALUES (?, ?, ?)',
            (session['user_id'], query_text, datetime.now().strftime('%Y-%m-%d'))
        )

        # Retrieve relevant sections based on semantic similarity
        sections = retrieval.get_relevant_sections(query_text)

        # Save retrieval results to RETRIEVAL_RESULT table
        for sec in sections:
            database.insert_db(
                'INSERT INTO RETRIEVAL_RESULT (query_id, section_id, similarity_score) VALUES (?, ?, ?)',
                (query_id, sec['section_id'], sec['score'])
            )

        # Generate answer from retrieved content
        answer_text = retrieval.generate_answer(query_text, sections)

        # Save answer to ANSWER table
        database.insert_db(
            'INSERT INTO ANSWER (query_id, answer_text) VALUES (?, ?)',
            (query_id, answer_text)
        )

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

    results = database.query_db('''
        SELECT rr.similarity_score, ds.section_text
        FROM RETRIEVAL_RESULT rr
        JOIN DOCUMENT_SECTION ds ON rr.section_id = ds.section_id
        WHERE rr.query_id = ?
        ORDER BY rr.similarity_score DESC
    ''', (query_id,))

    return render_template('answer.html',
                           query=query_row,
                           answer=answer_row,
                           results=results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
