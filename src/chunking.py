"""chunking.py — режем текст на куски с перекрытием.

Вынесено в отдельный модуль, потому что и ingest.py (индексация), и любой
будущий eval-скрипт должны резать текст ОДИНАКОВО — иначе результаты
эксперимента "поменял chunk_size" будет неверно сравнивать разные вещи.
"""
from . import config


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Режем строку на куски по `size` символов, сдвигаясь с шагом (size - overlap).

    Пример при size=500, overlap=100: чанк 2 начинается на 400 символов
    раньше конца чанка 1 — та же мысль, разорванная на границе, попадёт
    целиком хотя бы в один из двух чанков.
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks
