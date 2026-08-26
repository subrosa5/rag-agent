"""rag.py — grounded-ответ поверх retrieval: контекст + вопрос -> LLM -> ответ.

Это самостоятельный RAG (без агента) — можно гонять как есть, и это же
станет одним из инструментов агента в agent.py.
"""
from groq import Groq

from . import config
from .retrieval import hybrid_retrieve

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "Ты отвечаешь ТОЛЬКО на основе предоставленного контекста. "
    "Если ответа в контексте нет — прямо скажи, что не знаешь, не выдумывай. "
    "Если используешь факт из контекста — это ожидаемо и нормально, "
    "отвечай развёрнуто, но не добавляй ничего, чего нет в контексте."
)


def format_context(chunks: list[dict]) -> str:
    """Нумеруем источники — модель сможет на них ссылаться, и мы сможем
    показать пользователю, откуда взят ответ (важно для доверия к RAG)."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (источник: {c['source']})\n{c['content']}")
    return "\n\n".join(parts)


def answer(question: str, k: int = None) -> dict:
    chunks = hybrid_retrieve(question)
    if not chunks:
        return {"answer": "В базе документов пока ничего нет.", "sources": []}

    context = format_context(chunks)
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"

    resp = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"source": c["source"], "score": round(c["rerank_score"], 3)} for c in chunks],
    }


if __name__ == "__main__":
    q = input("Вопрос: ")
    result = answer(q)
    print("\n" + result["answer"])
    print("\n--- источники ---")
    for s in result["sources"]:
        print(f"  {s['source']}  (score={s['score']})")
