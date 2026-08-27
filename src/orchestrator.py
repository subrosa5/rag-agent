"""orchestrator.py — второй, независимый агент (критик) поверх основного.

Это MULTI-AGENT в буквальном смысле: не один агент с двумя инструментами
(как agent.py сам по себе), а ДВА независимых вызова модели с разными
ролями, где один проверяет другого — reflection-паттерн.

    вопрос → Agent → черновик ответа + trace (что искал, что нашёл)
                              │
                              ▼
                        Critic Agent
                    (сверяет ответ с trace)
                        ├─ OK → отдать пользователю
                        └─ REVISE: причина → Agent пробует снова (1 раз)
                                              с учётом замечания критика
                              │
                              ▼
                    отдать пользователю в любом случае
                    (после max_revisions критик не блокирует навсегда —
                     лучше вернуть неидеальный ответ, чем зависнуть)

Честный trade-off, который стоит проговорить: это удваивает-утраивает
число вызовов LLM на один вопрос — больше задержка, больше стоимость.
Осознанная цена за надёжность, не бесплатная фича.
"""
from groq import Groq

from . import config
from .agent import run_agent_verbose

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
    parts = []
    for step in trace:
        parts.append(f"вызвал {step['tool']}({step['args']}) → {step['result']}")
    return "\n".join(parts)


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
        temperature=0,  # критику нужна повторяемость, не креативность
    )
    verdict = resp.choices[0].message.content.strip()
    if verdict.upper().startswith("OK"):
        return True, ""
    return False, verdict


def run_agent_with_review(question: str, max_revisions: int = 1, verbose: bool = True) -> str:
    """Главная точка входа: агент отвечает, критик проверяет, при
    необходимости — одна попытка доработки."""
    result = run_agent_verbose(question, verbose=verbose)

    for attempt in range(max_revisions):
        ok, reason = critique(question, result["answer"], result["trace"])
        if verbose:
            status = "OK" if ok else f"REVISE ({reason})"
            print(f"  [критик] {status}")
        if ok:
            return result["answer"]
        # передаём замечание критика обратно агенту как инструкцию на доработку
        result = run_agent_verbose(question, extra_instruction=reason, verbose=verbose)

    # после max_revisions критик не блокирует ответ навсегда — лучше
    # вернуть неидеальный ответ пользователю, чем зависнуть в цикле проверок
    return result["answer"]


if __name__ == "__main__":
    q = input("Вопрос: ")
    print("\n" + run_agent_with_review(q))
