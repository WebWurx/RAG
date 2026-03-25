import os
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
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

        hashed = generate_password_hash(password)
        database.insert_db(
            'INSERT INTO USER (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed)
        )
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        user = database.query_db(
            'SELECT * FROM USER WHERE username = ?', (username,), one=True
        )
        if user and check_password_hash(user['password'], password):
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
    docs = database.query_db(
        'SELECT * FROM DOCUMENT WHERE user_id = ? ORDER BY upload_date DESC',
        (session['user_id'],)
    )
    return render_template('dashboard.html', docs=docs)


@app.route('/upload', methods=['POST'])
@login_required
def upload():
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

    # Save document record
    doc_id = database.insert_db(
        'INSERT INTO DOCUMENT (user_id, document_name, upload_date, file_type) VALUES (?, ?, ?, ?)',
        (session['user_id'], filename, datetime.now().strftime('%Y-%m-%d %H:%M'), file_type)
    )

    # Extract text and chunk with page tracking
    if file_type == 'pdf':
        pages = document_processor.extract_text_from_pdf(filepath)
        if not pages:
            flash('Could not extract text from the file.', 'danger')
            return redirect(url_for('dashboard'))
        chunks = document_processor.chunk_pdf_pages(pages)
    else:
        text = document_processor.extract_text_from_txt(filepath)
        if not text.strip():
            flash('Could not extract text from the file.', 'danger')
            return redirect(url_for('dashboard'))
        chunks = document_processor.chunk_text(text)

    for chunk_text, page_num in chunks:
        embedding_blob = retrieval.embed_to_blob(chunk_text)
        database.insert_db(
            'INSERT INTO DOCUMENT_SECTION (document_id, section_text, page_number, embedding) VALUES (?, ?, ?, ?)',
            (doc_id, chunk_text, page_num, embedding_blob)
        )

    flash(f'"{filename}" uploaded and processed successfully ({len(chunks)} sections).', 'success')

    return redirect(url_for('dashboard'))


@app.route('/delete/<int:document_id>', methods=['POST'])
@login_required
def delete_document(document_id):
    doc = database.query_db(
        'SELECT * FROM DOCUMENT WHERE document_id = ? AND user_id = ?',
        (document_id, session['user_id']), one=True
    )
    if not doc:
        flash('Document not found.', 'danger')
        return redirect(url_for('dashboard'))

    # Delete physical file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], doc['document_name'])
    if os.path.exists(filepath):
        os.remove(filepath)

    # Delete sections and cascade to retrieval results
    sections = database.query_db(
        'SELECT section_id FROM DOCUMENT_SECTION WHERE document_id = ?', (document_id,)
    )
    for section in sections:
        database.insert_db(
            'DELETE FROM RETRIEVAL_RESULT WHERE section_id = ?', (section['section_id'],)
        )
    database.insert_db('DELETE FROM DOCUMENT_SECTION WHERE document_id = ?', (document_id,))
    database.insert_db('DELETE FROM DOCUMENT WHERE document_id = ?', (document_id,))

    flash(f'"{doc["document_name"]}" deleted successfully.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/document/<int:document_id>/preview')
@login_required
def document_preview(document_id):
    doc = database.query_db(
        'SELECT * FROM DOCUMENT WHERE document_id = ? AND user_id = ?',
        (document_id, session['user_id']), one=True
    )
    if not doc:
        return jsonify({'error': 'Not found'}), 404

    sections = database.query_db(
        'SELECT section_text, page_number FROM DOCUMENT_SECTION WHERE document_id = ? ORDER BY page_number, section_id',
        (document_id,)
    )
    return jsonify({
        'name': doc['document_name'],
        'sections': [{'text': s['section_text'], 'page': s['page_number']} for s in sections]
    })


@app.route('/query', methods=['GET', 'POST'])
@login_required
def query():
    if request.method == 'POST':
        query_text = request.form['query_text'].strip()
        if not query_text:
            flash('Please enter a question.', 'danger')
            return render_template('query.html')

        # Save query
        query_id = database.insert_db(
            'INSERT INTO QUERY (user_id, query_text, query_date) VALUES (?, ?, ?)',
            (session['user_id'], query_text, datetime.now().strftime('%Y-%m-%d %H:%M'))
        )

        # Retrieve relevant sections
        sections = retrieval.get_relevant_sections(query_text, top_k=8)

        # Save retrieval results
        for sec in sections:
            database.insert_db(
                'INSERT INTO RETRIEVAL_RESULT (query_id, section_id, similarity_score) VALUES (?, ?, ?)',
                (query_id, sec['section_id'], sec['score'])
            )

        # Generate answer
        answer_text = retrieval.generate_answer(query_text, sections)

        # Save answer
        database.insert_db(
            'INSERT INTO ANSWER (query_id, answer_text) VALUES (?, ?)',
            (query_id, answer_text)
        )

        return redirect(url_for('answer', query_id=query_id))

    return render_template('query.html')


@app.route('/answer/<int:query_id>')
@login_required
def answer(query_id):
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
        SELECT rr.similarity_score, ds.section_text, ds.page_number, d.document_name
        FROM RETRIEVAL_RESULT rr
        JOIN DOCUMENT_SECTION ds ON rr.section_id = ds.section_id
        JOIN DOCUMENT d ON ds.document_id = d.document_id
        WHERE rr.query_id = ?
        ORDER BY rr.similarity_score DESC
    ''', (query_id,))

    # Build compact source citations: {doc_name: sorted list of page numbers}
    sources = {}
    for r in results:
        name = r['document_name']
        if name not in sources:
            sources[name] = set()
        sources[name].add(r['page_number'])
    sources = {name: sorted(pages) for name, pages in sources.items()}

    return render_template('answer.html',
                           query=query_row,
                           answer=answer_row,
                           sources=sources)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
