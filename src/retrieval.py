"""retrieval.py — hybrid search (vector + BM25) с реранкингом.

Труба (важно уметь нарисовать это на доске):

  вопрос
    │
    ├── vector_search  (семантика: "похоже по смыслу")   ─┐
    │                                                      ├─> union по id ─> rerank (cross-encoder) ─> top-N
    └── bm25_search     (лексика: "совпадают слова/термины") ┘

Зачем оба, а не один векторный поиск:
эмбеддинги отлично ловят перефразировки ("сколько стоит" ~ "какая цена"),
но плохо ловят точные токены — имена, коды ошибок, версии библиотек,
которых модель эмбеддинга могла не увидеть в достаточном объёме при обучении.
BM25 — классический lexical-поиск (term frequency), он эти токены найдёт
всегда, даже если семантически предложение выглядит по-другому.

Зачем реранкинг поверх union, а не просто взять top-k от каждого:
bi-encoder (наша модель эмбеддингов) кодирует вопрос и чанк НЕЗАВИСИМО друг
от друга — это и делает его быстрым (можно проиндексировать миллион чанков
заранее), но менее точным. Cross-encoder читает пару (вопрос, чанк) ВМЕСТЕ
и выдаёт единый score релевантности — точнее, но дорого считать на всей базе.
Поэтому: bi-encoder для recall (быстро, грубо, много кандидатов),
cross-encoder для precision (медленно, точно, мало кандидатов).
"""
from rank_bm25 import BM25Okapi

from . import config, db
from .embeddings import embed, get_reranker


def vector_search(question: str, k: int = None) -> list[dict]:
    k = k or config.VECTOR_TOP_K
    q_vec = embed([question])[0]
    return db.vector_search(q_vec, k)


def bm25_search(question: str, k: int = None) -> list[dict]:
    """Наивная реализация: перечитываем все чанки и строим BM25-индекс на
    каждый запрос. Ок для сотен-тысяч чанков в лабе. В проде BM25-индекс
    (или Postgres full-text search, см. gin-индекс в schema.sql) строится
    один раз при индексации и переиспользуется, а не пересчитывается на
    каждый вопрос."""
    k = k or config.BM25_TOP_K
    all_chunks = db.fetch_all_chunks()
    if not all_chunks:
        return []

    tokenized_corpus = [c["content"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(all_chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [{**c, "score": float(s)} for c, s in ranked[:k]]


def hybrid_retrieve(question: str) -> list[dict]:
    """Объединяем кандидатов из обоих поисков (без дублей по id), потом реранкаем."""
    vector_hits = vector_search(question)
    lexical_hits = bm25_search(question)

    by_id = {}
    for hit in vector_hits + lexical_hits:
        by_id[hit["id"]] = hit  # если чанк нашли оба поиска — оставляем одну запись
    candidates = list(by_id.values())

    if not candidates:
        return []
    return rerank(question, candidates)


def rerank(question: str, candidates: list[dict], top_k: int = None) -> list[dict]:
    top_k = top_k or config.RERANK_TOP_K
    reranker = get_reranker()

    pairs = [[question, c["content"]] for c in candidates]
    scores = reranker.predict(pairs)  # cross-encoder: (вопрос, чанк) -> релевантность

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
