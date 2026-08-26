"""embeddings.py — единая точка загрузки модели эмбеддингов и реранкера.

КЛЮЧЕВОЙ момент, который любят спрашивать на интервью: вопрос и чанки
ДОЛЖНЫ эмбеддиться одной и той же моделью. Если ingest.py и retrieval.py
каждый по отдельности создают SentenceTransformer(...), легко словить баг
(кто-то поменял модель в одном месте и забыл в другом) и получить мусорный
поиск без единой ошибки в логах. Поэтому модель живёт в одном модуле.

Модели грузятся лениво (при первом обращении) и кэшируются в памяти процесса —
загрузка весов занимает секунды, не хотим платить это на каждый вызов.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder

from . import config


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL)


def embed(texts: list[str]):
    """Список строк -> numpy-массив векторов. normalize_embeddings=True делает
    векторы единичной длины, тогда dot product == cosine similarity — pgvector
    считает это под капотом через оператор <=>, но полезно понимать, почему
    нормализация вообще нужна."""
    return get_embedder().encode(texts, normalize_embeddings=True)
