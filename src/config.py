"""config.py — все настраиваемые параметры в одном месте.

Зачем отдельный файл, а не константы прямо в коде: когда параметр
(модель, k, размер чанка) разбросан по 5 файлам, эксперимент "а что если
увеличить top_k" превращается в grep по всему проекту. Здесь — одна точка правки.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # читает .env в корне проекта, если он есть

# --- LLM ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# llama-3.1-8b-instant (из большинства туториалов) Groq сняли с линейки —
# поймано прямо здесь, при первом реальном вызове. openai/gpt-oss-20b:
# open-weight модель от OpenAI, хостится на Groq, быстрая и поддерживает
# tool calling — нужно для agent.py.
LLM_MODEL = "openai/gpt-oss-20b"

# --- эмбеддинги и реранкинг: два бэкенда за одним интерфейсом (src/embeddings.py) ---
#
# local  — sentence-transformers, модели крутятся в этом же процессе.
#          Бесплатно, приватно, но обе модели вместе ~1.3GB RAM — ок на
#          твоей машине, но Render free/starter tier (512MB) падает с OOM.
# cohere — Cohere Embed + Rerank API. Модели не грузятся в память вообще,
#          это HTTP-вызов наружу. Компромисс обратный: почти нулевая память
#          на нашей стороне, но платишь за вызов (у Cohere есть бесплатный
#          trial-лимит) и добавляется сетевая задержка на каждый запрос.
#
# Переключается переменной окружения USE_COHERE. Локально (по умолчанию) —
# local, на Render — cohere. Само переключение объясняет хороший вопрос с
# интервью: "как ты выбираешь self-hosted vs managed inference?" — ответ
# здесь буквально закодирован как явный trade-off, а не решён втихую.
USE_COHERE = os.environ.get("USE_COHERE", "false").lower() == "true"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if USE_COHERE:
    EMBEDDING_MODEL = "embed-multilingual-v3.0"
    RERANKER_MODEL = "rerank-multilingual-v3.0"
    EMBEDDING_DIM = 1024   # у Cohere другая размерность вектора, чем у локальной модели —
                           # ЭТО ВАЖНО: таблица в базе должна быть создана под то EMBEDDING_DIM,
                           # которое реально используется (см. db.py::init_schema).
else:
    # ВАЖНО: all-MiniLM-L6-v2 (дефолт из большинства туториалов) обучен почти
    # только на английском — на русских документах он путает вообще всё
    # семантически несвязанное, потому что не различает нюансы. Проверено на
    # практике в этом же проекте: с ним нужный чанк про откат деплоя падал на
    # последнее место при явно релевантном вопросе.
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    EMBEDDING_DIM = 384

# --- чанкинг ---
CHUNK_SIZE = 500       # символов, не токенов — упрощение ради понятности; в проде считают токенами
CHUNK_OVERLAP = 100    # 20% — не даёт потерять мысль, разорванную на границе чанка

# --- retrieval ---
VECTOR_TOP_K = 20       # сколько кандидатов берём из векторного поиска (recall stage)
RERANK_TOP_K = 4        # сколько оставляем после реранкинга (precision stage) — это уйдёт в промпт
BM25_TOP_K = 20          # столько же кандидатов берём из lexical-поиска для hybrid

# --- база данных ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/rag_agent")
