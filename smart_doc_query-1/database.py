import sqlite3
import config


def get_db():
    conn = sqlite3.connect(config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the 6 tables as specified in the Design Phase document (pages 9-10)."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.executescript('''
        -- Table 1: USER (Design Phase, Page 9)
        -- Fields: user_id, username, email, password
        CREATE TABLE IF NOT EXISTS USER (
            user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            username  VARCHAR(100) NOT NULL UNIQUE,
            email     VARCHAR(100) NOT NULL UNIQUE,
            password  VARCHAR(100) NOT NULL
        );

        -- Table 2: DOCUMENT (Design Phase, Page 9)
        -- Fields: document_id, user_id, document_name, upload_date, file_type
        CREATE TABLE IF NOT EXISTS DOCUMENT (
            document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            document_name VARCHAR(150) NOT NULL,
            upload_date   DATE NOT NULL,
            file_type     VARCHAR(20) NOT NULL,
            FOREIGN KEY (user_id) REFERENCES USER(user_id)
        );

        -- Table 3: DOCUMENT_SECTION (Design Phase, Page 9)
        -- Fields: section_id, document_id, section_text
        CREATE TABLE IF NOT EXISTS DOCUMENT_SECTION (
            section_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id  INTEGER NOT NULL,
            section_text TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES DOCUMENT(document_id)
        );

        -- Table 4: QUERY (Design Phase, Page 10)
        -- Fields: query_id, user_id, query_text, query_date
        CREATE TABLE IF NOT EXISTS QUERY (
            query_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            query_date DATE NOT NULL,
            FOREIGN KEY (user_id) REFERENCES USER(user_id)
        );

        -- Table 5: RETRIEVAL_RESULT (Design Phase, Page 10)
        -- Fields: result_id, query_id, section_id, similarity_score
        CREATE TABLE IF NOT EXISTS RETRIEVAL_RESULT (
            result_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id         INTEGER NOT NULL,
            section_id       INTEGER NOT NULL,
            similarity_score FLOAT NOT NULL,
            FOREIGN KEY (query_id)   REFERENCES QUERY(query_id),
            FOREIGN KEY (section_id) REFERENCES DOCUMENT_SECTION(section_id)
        );

        -- Table 6: ANSWER (Design Phase, Page 10)
        -- Fields: answer_id, query_id, answer_text
        CREATE TABLE IF NOT EXISTS ANSWER (
            answer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id    INTEGER NOT NULL,
            answer_text TEXT NOT NULL,
            FOREIGN KEY (query_id) REFERENCES QUERY(query_id)
        );
    ''')

    conn.commit()
    conn.close()


def query_db(sql, args=(), one=False):
    conn = get_db()
    cursor = conn.execute(sql, args)
    rows = cursor.fetchall()
    conn.close()
    return (rows[0] if rows else None) if one else rows


def insert_db(sql, args=()):
    conn = get_db()
    cursor = conn.execute(sql, args)
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id
