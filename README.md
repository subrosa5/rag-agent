# rag-agent

RAG + tool-use агент, реализованный без фреймворков (без LangChain/LlamaIndex) —
hybrid retrieval, реранкинг, tool-calling цикл и eval-харнес написаны напрямую
поверх Postgres/pgvector, Groq и Cohere.

**Демо:** https://rag-agent-liard-two.vercel.app
**Стек:** Next.js (Vercel) → FastAPI (Render) → Postgres/pgvector (Neon), LLM и реранкинг через Groq/Cohere

## Что делает

Отвечает на вопросы по набору документов (RAG) и умеет вызывать инструменты —
сейчас это поиск по базе знаний и калькулятор. Модель сама решает, какой
инструмент вызвать и сколько раз, вместо жёсткого pipeline "всегда искать,
потом отвечать".

Пример: на вопрос *"Сколько времени занимает откат деплоя и сколько будет 12×7?"*
агент вызывает `search_documents` для первой части и `calculate` для второй,
и собирает оба результата в один ответ.

## Архитектура

```
                         ┌──────────────────┐
   вопрос ───────────────▶      AGENT       │◀── история диалога, tool results
                         │ (Groq, gpt-oss-20b)│
                         └────────┬─────────┘
                                  │ модель решает: нужен инструмент?
                     ┌────────────┴────────────┐
                     ▼                         ▼
            search_documents             calculate
                     │
        ┌────────────┴─────────────┐
        ▼                           ▼
  vector_search               bm25_search
  (pgvector, HNSW,             (rank_bm25,
   косинус)                    lexical)
        └────────────┬─────────────┘
                      ▼
              union по id
                      ▼
         rerank (cross-encoder,
         top-20 → top-4)
                      ▼
          чанки с источниками ──▶ обратно в историю агента ──▶ финальный ответ
```

## Ключевые решения

- **pgvector вместо отдельной vector DB.** Postgres с HNSW-индексом достаточен
  вплоть до миллионов векторов; отдельная инфраструктура не нужна для этого масштаба.
- **Hybrid search (vector + BM25).** Эмбеддинги ловят смысл, но плохо ловят
  точные токены — версии, коды, имена. BM25 закрывает этот случай. Кандидаты
  из обоих источников объединяются и фильтруются реранкингом.
- **Двухэтапный retrieval (bi-encoder → cross-encoder).** Bi-encoder кодирует
  вопрос и документ независимо — быстро, подходит для recall по всей базе.
  Cross-encoder читает пару вместе — точнее, но дорого считать на всей базе,
  поэтому применяется только к уже отобранным top-20 кандидатам.
- **RAG как инструмент агента, а не отдельный pipeline.** `rag.py` — фиксированный
  retrieve→generate; `agent.py` — модель сама решает, вызывать ли `search_documents`,
  `calculate`, оба или ни одного.
- **Калькулятор через `ast`, не `eval()`.** LLM генерирует выражение как строку;
  прямой `eval` — инъекция. `ast`-парсер разрешает только арифметику.
- **Два бэкенда эмбеддингов/реранкинга (local / Cohere), переключаемые `USE_COHERE`.**
  Self-hosted sentence-transformers держат в памяти ~1.3GB (обе модели) — не
  помещается в 512MB на Render free/starter tier. Cohere Embed+Rerank API
  переносит инференс наружу, локальный процесс почти ничего не занимает.
  Остальной код (`retrieval.py`, `ingest.py`) не знает, какой бэкенд активен.

## Известные ограничения

- **Без памяти между сообщениями.** Каждый запрос к `/agent/ask` независим —
  фронт не отправляет историю диалога, `agent.py` не хранит сессию. Агент не
  помнит предыдущий вопрос в рамках одного чата.
- **Eval-набор маленький** (4 вопроса) — показывает метод, не даёт
  статистической значимости.
- **LLM-судья в eval неидеален**: на прогоне против Neon судья иногда
  засчитывает FAIL честному отказу ("не знаю") на out-of-scope вопрос, хотя
  собственный промпт судьи требует засчитывать это как PASS — сам судья
  требует отдельной валидации.
- **Без авторизации** — API открыт всем, кто знает URL.

## Структура

```
rag-agent/
├── data/*.txt              # документы для индексации
├── db/schema.sql            # справочная структура таблицы chunks (реально создаётся динамически, db.py)
├── render.yaml               # Render Blueprint
├── web/                        # Next.js чат-фронт (отдельный деплой на Vercel)
├── src/
│   ├── config.py         # параметры, переключатель local/Cohere бэкенда
│   ├── db.py              # слой поверх psycopg, без ORM
│   ├── chunking.py        # текст -> чанки с overlap
│   ├── embeddings.py      # эмбеддинги/реранкинг: local (sentence-transformers) или Cohere API
│   ├── ingest.py           # data/*.txt -> чанки -> эмбеддинги -> Postgres
│   ├── retrieval.py        # hybrid search + rerank
│   ├── rag.py               # retrieve + generate, без агента
│   ├── tools.py              # search_documents, calculate
│   ├── agent.py               # tool-use цикл поверх Groq
│   ├── eval.py                 # retrieval hit rate + faithfulness (LLM-judge)
│   └── api.py                   # FastAPI: /agent/ask, /rag/ask
├── requirements.txt         # база, без torch (то, что ставится на Render)
└── requirements-local.txt    # + sentence-transformers, для self-hosted режима
```

## Локальный запуск

```bash
cp .env.example .env   # заполнить GROQ_API_KEY; USE_COHERE=false по умолчанию

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt

python -m src.ingest              # индексация data/*.txt

python -m src.rag                 # RAG без агента
python -m src.agent               # агент с инструментами
python -m src.eval                # eval-метрики

uvicorn src.api:app --reload      # HTTP API
```

## Деплой

Прод использует Cohere вместо self-hosted моделей и Neon вместо локального Postgres.

1. **Neon** — Postgres с pgvector, провизионится через Vercel Marketplace
   (`vercel integration add neon`) или напрямую на neon.tech.
2. **Render** — подключить GitHub-репозиторий, он подхватит `render.yaml`.
   Секреты задаются в дашборде: `GROQ_API_KEY`, `COHERE_API_KEY`,
   `DATABASE_URL` (из Neon), `ALLOWED_ORIGIN` (домен фронта).
3. `python -m src.ingest` один раз против `DATABASE_URL` от Neon — наполняет базу.
4. **Vercel** — `web/` деплоится отдельным проектом, env `RAG_AGENT_API_URL`
   указывает на Render-сервис.

## Метрики

Прогон `eval.py` против прод-базы: retrieval hit rate 100%, faithfulness 75%
(4 вопроса; единственный "провал" — ложное срабатывание судьи, см. "Известные ограничения").

## Roadmap

- История диалога (память между сообщениями)
- Golden set 30–50 вопросов вместо 4
- Streaming ответа агента
- Авторизация API
- Multi-tenant (`tenant_id`, изоляция документов между организациями)
