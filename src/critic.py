"""critic.py — агент, проверяющий ответ основного агента ПЕРЕД тем, как его
увидит пользователь (reflection-паттерн).

Роль отличается от guardrail.py: guardrail проверяет ВХОД (безопасен ли
вопрос), критик проверяет ВЫХОД (обоснован ли ответ тем, что реально нашёл
агент). Разная работа, разная цена ошибки — поэтому разные агенты.
"""
from groq import Groq

from . import config

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


CRITIC_SYSTEM_PROMPT = (
    "Ты — критик, проверяющий ответы другого AI-агента перед тем, как их "
    "увидит пользователь. Тебе даны: вопрос пользователя, черновик ответа, "
    "и trace — что агент искал и что нашёл (если искал вообще). "
    "Проверь:\n"
    "1) Ответ действительно отвечает на вопрос?\n"
    "2) Если агент использовал search_documents — ответ следует из найденного, "
    "без придуманных фактов?\n"
    "3) Если агент НЕ нашёл ничего релевантного — сказал ли он честно "
    "'не знаю', вместо того чтобы выдумать?\n\n"
    "Ответь строго в формате:\n"
    "OK — если всё в порядке\n"
    "REVISE: <короткая причина> — если нужно исправить"
)


def _format_trace(trace: list[dict]) -> str:
    if not trace:
        return "(агент не вызывал инструменты — ответил из общих знаний)"
    return "\n".join(
        f"вызвал {step['tool']}({step['args']}) → {step['result']}" for step in trace
    )


def critique(question: str, answer: str, trace: list[dict]) -> tuple[bool, str]:
    """Возвращает (ok, причина). ok=True — критик пропускает ответ как есть."""
    prompt = (
        f"Вопрос пользователя: {question}\n\n"
        f"Что делал агент:\n{_format_trace(trace)}\n\n"
        f"Черновик ответа агента: {answer}\n\n"
        f"Вердикт:"
    )
    resp = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip()
    if verdict.upper().startswith("OK"):
        return True, ""
    # тот же приём, что в guardrail.py — не дублировать префикс "REVISE:" в логе
    reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
    return False, reason
