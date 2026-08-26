"""embeddings.py — эмбеддинги и реранкинг за одним интерфейсом, два бэкенда.

Остальной код (ingest.py, retrieval.py) вызывает embed() и rerank() и не
знает и не должен знать, крутится ли модель локально или это HTTP-вызов
к Cohere — это и есть смысл вынести бэкенд за интерфейс. Какой бэкенд
активен, решает config.USE_COHERE (env-переменная), объяснение выбора — там.
"""
from functools import lru_cache

import numpy as np

from . import config


if config.USE_COHERE:
    import cohere

    @lru_cache(maxsize=1)
    def _client() -> "cohere.ClientV2":
        return cohere.ClientV2(api_key=config.COHERE_API_KEY)

    def embed(texts: list[str], input_type: str = "search_document"):
        """input_type различает "это чанк документа для индекса" и "это вопрос
        пользователя" — Cohere кодирует их немного по-разному внутри модели,
        это часть их API. У локальной bi-encoder модели такого разделения нет —
        вопрос и документ эмбеддятся одинаково, поэтому параметр там просто
        игнорируется (см. ветку else ниже)."""
        resp = _client().embed(
            texts=texts,
            model=config.EMBEDDING_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        # Cohere отдаёт обычные python list[float], а не numpy-массив. pgvector'овский
        # register_vector умеет сериализовать в тип `vector` только numpy.ndarray —
        # с обычным списком psycopg молча сериализует его как Postgres double
        # precision[], и запрос падает ("vector <=> double precision[]").
        # Поймано на реальном Neon, локально не проявлялось (local-бэкенд и так
        # возвращает ndarray из sentence-transformers).
        return np.array(resp.embeddings.float_)

    def rerank(question: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        top_k = top_k or config.RERANK_TOP_K
        if not candidates:
            return []
        docs = [c["content"] for c in candidates]
        resp = _client().rerank(
            model=config.RERANKER_MODEL,
            query=question,
            documents=docs,
            top_n=top_k,
        )
        out = []
        for r in resp.results:
            c = candidates[r.index]
            c["rerank_score"] = r.relevance_score
            out.append(c)
        return out

else:
    from sentence_transformers import SentenceTransformer, CrossEncoder

    @lru_cache(maxsize=1)
    def _embedder() -> SentenceTransformer:
        return SentenceTransformer(config.EMBEDDING_MODEL)

    @lru_cache(maxsize=1)
    def _reranker() -> CrossEncoder:
        return CrossEncoder(config.RERANKER_MODEL)

    def embed(texts: list[str], input_type: str = "search_document"):
        # normalize_embeddings=True -> единичная длина вектора, тогда dot product
        # == cosine similarity; pgvector сам считает косинус через <=>, но полезно
        # понимать, откуда нормализация вообще берётся.
        return _embedder().encode(texts, normalize_embeddings=True)

    def rerank(question: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        top_k = top_k or config.RERANK_TOP_K
        if not candidates:
            return []
        pairs = [[question, c["content"]] for c in candidates]
        scores = _reranker().predict(pairs)  # cross-encoder: (вопрос, чанк) -> релевантность
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]
