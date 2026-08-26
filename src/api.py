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
from pydantic import BaseModel

from .agent import run_agent
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


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/agent/ask", response_model=AskResponse)
def ask_agent(req: AskRequest):
    """Полный агент с доступом к инструментам (RAG + калькулятор)."""
    return {"answer": run_agent(req.question, verbose=False)}


@app.post("/rag/ask")
def ask_rag(req: AskRequest):
    """Чистый RAG без агентного цикла — для сравнения и для случаев,
    когда нужен именно фиксированный pipeline, а не решение модели."""
    return rag_answer(req.question)


@app.get("/health")
def health():
    return {"status": "ok"}
