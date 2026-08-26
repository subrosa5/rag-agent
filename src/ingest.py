"""ingest.py — строит индекс: файлы из data/ -> чанки -> эмбеддинги -> Postgres.

Запуск: python -m src.ingest
"""
import glob
import os

from . import config, db
from .chunking import chunk_text
from .embeddings import embed


def load_documents(data_dir: str = "data") -> list[tuple[str, str]]:
    """Возвращает [(путь_к_файлу, содержимое), ...]."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append((path, f.read()))
    return docs


def run():
    docs = load_documents()
    if not docs:
        print("Нет файлов в data/*.txt — добавь документы и запусти снова.")
        return

    print(f"Найдено документов: {len(docs)}")
    db.init_schema()   # идемпотентно: создаст таблицу/индекс, если их ещё нет (новая база — Neon и т.п.)
    db.clear_chunks()  # полная переиндексация — проще инкрементальной синхронизации

    total = 0
    for source, text in docs:
        pieces = chunk_text(text)
        if not pieces:
            continue
        vectors = embed(pieces)  # одним батчем — быстрее, чем по одному чанку
        rows = [
            (source, i, chunk, vec)
            for i, (chunk, vec) in enumerate(zip(pieces, vectors))
        ]
        db.insert_chunks(rows)
        total += len(rows)
        print(f"  {source}: {len(pieces)} чанков")

    print(f"Готово: {total} чанков проиндексировано в Postgres.")


if __name__ == "__main__":
    run()
