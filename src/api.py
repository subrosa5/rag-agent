"""api.py — тонкий HTTP-слой поверх агента.

Запуск: uvicorn src.api:app --reload
Зачем: "production-ready" в резюме проекта значит, что систему можно
дёрнуть не только из терминала. Сам агент (agent.py) от FastAPI не зависит —
это намеренно: логика и транспорт разделены, api.py можно заменить на
CLI/Slack-бота/что угодно, не трогая agent.py.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from .agent import run_agent
from .rag import answer as rag_answer

app = FastAPI(title="RAG + Agent lab")


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
