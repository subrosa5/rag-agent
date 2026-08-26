"""db.py — тонкий слой поверх psycopg. Никакого ORM.

Почему без ORM (SQLAlchemy и т.п.): для 3 запросов ORM — это лишний слой
абстракции, который на интервью сложнее объяснить, чем сам SQL. Здесь ты
видишь ровно тот запрос, который уходит в базу.
"""
import psycopg
from pgvector.psycopg import register_vector
from contextlib import contextmanager

from . import config


@contextmanager
def get_conn():
    """Контекстный менеджер: открыли соединение, зарегистрировали тип vector, отдали, закрыли."""
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    register_vector(conn)  # без этого psycopg не знает, как сериализовать numpy-массив в vector(384)
    try:
        yield conn
    finally:
        conn.close()


def insert_chunks(rows: list[tuple[str, int, str, "np.ndarray"]]) -> None:
    """rows: список (source, chunk_index, content, embedding)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (source, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s)""",
                rows,
            )


def clear_chunks() -> None:
    """Полная переиндексация проще, чем инкрементальные апдейты — для лабы этого достаточно."""
    with get_conn() as conn:
        conn.execute("TRUNCATE chunks RESTART IDENTITY")


def vector_search(query_embedding, k: int) -> list[dict]:
    """Поиск k ближайших чанков по косинусному расстоянию.

    Оператор <=> — косинусное расстояние (1 - cosine_similarity), встроен pgvector.
    ORDER BY <=> ASC значит "сначала самые близкие" (расстояние маленькое = похоже).
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, source, content, 1 - (embedding <=> %s) AS score
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (query_embedding, query_embedding, k),
        ).fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2], "score": r[3]} for r in rows]


def fetch_all_chunks() -> list[dict]:
    """Нужен для BM25-части hybrid search — rank_bm25 живёт в памяти, не в базе."""
    with get_conn() as conn:
        rows = conn.execute("SELECT id, source, content FROM chunks").fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2]} for r in rows]
