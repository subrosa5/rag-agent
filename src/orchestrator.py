"""orchestrator.py — собирает четырёх независимых агентов в один пайплайн.

    вопрос
      │
      ▼
   Guardrail Agent ──BLOCK──▶ вежливый отказ, дальше не идём (экономим вызовы)
      │ ALLOW
      ▼
   Query Rewriter Agent  (сырой вопрос → чистая формулировка)
      │
      ▼
   Main Agent (agent.py) — решает сам: search_documents / calculate / оба
      │
      ▼
   Critic Agent ──REVISE──▶ Main Agent пробует снова (максимум 1 раз)
      │ OK
      ▼
   ответ пользователю

Четыре роли, у каждой свой периметр ответственности — это то, что делает
это НАСТОЯЩИМ multi-agent, а не косметикой:
  Guardrail       — безопасность на входе (защита от prompt injection)
  Query Rewriter  — качество поискового запроса (реальный RAG-приём)
  Main Agent      — выполнение задачи через инструменты
  Critic          — обоснованность ответа на выходе

Честная цена: до 5 вызовов LLM на один вопрос в худшем случае (guardrail +
rewriter + agent + critic + 1 повтор agent'а). Для игрушечного демо это
избыточно; для системы, которая не должна врать и не должна вестись на
инъекции, — осознанный и оправданный trade-off.
"""
from .agent import run_agent_verbose
from .critic import critique
from .guardrail import check as guardrail_check
from .memory import trim_history
from .query_rewriter import rewrite as rewrite_query


def run_agent_with_review(
    question: str,
    history: list[dict] = None,
    max_revisions: int = 1,
    verbose: bool = True,
) -> str:
    history = trim_history(history or [])  # см. memory.py — иначе контекст растёт бесконечно

    allowed, reason = guardrail_check(question)
    if verbose:
        print(f"  [guardrail] {'ALLOW' if allowed else 'BLOCK: ' + reason}")
    if not allowed:
        return "Извините, не могу обработать этот запрос."

    clean_question = rewrite_query(question, history=history)
    if verbose and clean_question != question:
        print(f"  [rewriter] '{question}' → '{clean_question}'")

    result = run_agent_verbose(clean_question, history=history, verbose=verbose)

    for _ in range(max_revisions):
        ok, crit_reason = critique(question, result["answer"], result["trace"])
        if verbose:
            print(f"  [критик] {'OK' if ok else 'REVISE (' + crit_reason + ')'}")
        if ok:
            return result["answer"]
        # передаём замечание критика обратно агенту как инструкцию на доработку
        result = run_agent_verbose(
            clean_question, history=history, extra_instruction=crit_reason, verbose=verbose
        )

    # после max_revisions критик не блокирует ответ навсегда — лучше вернуть
    # неидеальный ответ, чем зависнуть в бесконечных проверках
    return result["answer"]


if __name__ == "__main__":
    q = input("Вопрос: ")
    print("\n" + run_agent_with_review(q))
