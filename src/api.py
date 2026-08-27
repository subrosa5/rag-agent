"""api.py — тонкий HTTP-слой поверх агента.

Запуск: uvicorn src.api:app --reload
Зачем: "production-ready" в резюме проекта значит, что систему можно
дёрнуть не только из терминала. Сам агент (agent.py) от FastAPI не зависит —
это намеренно: логика и транспорт разделены, api.py можно заменить на
CLI/Slack-бота/что угодно, не трогая agent.py.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import run_agent
from .orchestrator import run_agent_with_review
from .rag import answer as rag_answer

app = FastAPI(title="RAG + Agent lab")

# Фронт (Next.js на Vercel) и бэкенд (Render) — разные домены, значит браузер
# будет делать cross-origin запрос, и без CORS-заголовков он его заблокирует.
# ALLOWED_ORIGIN — env-переменная, а не хардкод домена, чтобы не редактировать
# код при каждом передеплое фронта на новый preview-URL.
_allowed_origin = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin] if _allowed_origin != "*" else ["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    # min_length=1 отсекает пустую строку прямо на уровне валидации запроса —
    # без этого пустой вопрос доходил бы до retrieval/Groq и тратил впустую
    # вызов внешнего API. FastAPI сам вернёт 422 с понятной ошибкой, наш код
    # даже не запустится.
    question: str = Field(min_length=1)
    # history приходит с фронта целиком (см. web/app/Chat.tsx) — фронт не
    # обрезает её сам, обрезка под контекстное окно — забота бэкенда
    # (src/memory.py), а не каждого клиента по отдельности.
    history: list[HistoryMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str


@app.post("/agent/ask", response_model=AskResponse)
def ask_agent(req: AskRequest):
    """Агент + критик (multi-agent, reflection-паттерн): второй независимый
    вызов LLM проверяет ответ первого перед тем, как отдать пользователю.
    Дороже и медленнее одного агента (см. orchestrator.py), но это то, что
    реально отвечает на фронте — сознательный выбор в пользу надёжности."""
    history = [m.model_dump() for m in req.history]
    return {"answer": run_agent_with_review(req.question, history=history, verbose=False)}


@app.post("/agent/ask-raw", response_model=AskResponse)
def ask_agent_raw(req: AskRequest):
    """Тот же агент, но БЕЗ критика — один вызов вместо двух-трёх. Для
    сравнения задержки/поведения и как честный пример, что multi-agent —
    не бесплатная надстройка, а конкретный trade-off надёжность/стоимость."""
    history = [m.model_dump() for m in req.history]
    return {"answer": run_agent(req.question, history=history, verbose=False)}


@app.post("/rag/ask")
def ask_rag(req: AskRequest):
    """Чистый RAG без агентного цикла — для сравнения и для случаев,
    когда нужен именно фиксированный pipeline, а не решение модели."""
    return rag_answer(req.question)


@app.get("/health")
def health():
    return {"status": "ok"}
