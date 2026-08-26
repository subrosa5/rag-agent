-- schema.sql — структура хранилища для RAG.
--
-- Почему одна таблица, а не "документы" + "чанки" в разных таблицах:
-- для нашего масштаба (десятки-сотни документов) это ненужное усложнение.
-- В реальном проекте с миллионами документов чанки хранят отдельно от
-- метаданных документа, но здесь важнее видеть механику, а не ORM-абстракции.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,          -- путь к исходному файлу — нужно для цитирования
    chunk_index INTEGER NOT NULL,       -- номер чанка внутри документа (для отладки/дебага)
    content     TEXT NOT NULL,          -- сам текст чанка — это то, что уйдёт в промпт
    embedding   vector(384) NOT NULL,   -- 384 = размерность all-MiniLM-L6-v2. Меняешь модель — меняй размерность.
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW-индекс: приближённый поиск ближайших соседей.
-- Без индекса Postgres на каждый запрос делает full scan + считает косинус
-- для КАЖДОЙ строки — O(n). На тысячах чанков это уже заметно медленно.
-- HNSW строит граф "ближайших соседей" заранее, поиск становится ~O(log n).
-- vector_cosine_ops — говорим индексу, что мы будем искать по косинусному расстоянию
-- (должно совпадать с оператором, которым ищем в запросах: <=>).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Обычный текстовый индекс пригодится для BM25/lexical части hybrid search,
-- если решишь искать точные термины через Postgres full-text search вместо rank_bm25 в Python.
CREATE INDEX IF NOT EXISTS chunks_content_trgm_idx ON chunks USING gin (to_tsvector('english', content));
