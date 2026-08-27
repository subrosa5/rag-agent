"""agent.py — многошаговый агент с tool use поверх Groq.

Отличие от "игрушечного" агента (один tool call и всё) — здесь настоящий
цикл ReAct-стиля: модель может вызывать инструменты НЕСКОЛЬКО раз подряд,
опираясь на результаты предыдущих вызовов, пока сама не решит, что готова
ответить текстом.

    while шаг < max_steps:
        спросить модель
        если модель хочет tool_call(ы) -> выполнить, добавить результаты в историю, повторить
        если модель вернула текст -> это финальный ответ, выходим

Это и есть формальное отличие агента от RAG:
RAG — фиксированный pipeline (retrieve -> generate), шаг всегда один и тот же.
Агент — модель сама решает, сколько шагов сделать и какие инструменты вызвать.
"""
import json

from groq import Groq

from . import config
from .tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

SYSTEM_PROMPT = (
    "Ты полезный ассистент с доступом к инструментам: поиск по документам "
    "пользователя и калькулятор. Вызывай инструмент, когда он нужен для точного "
    "ответа. Если для ответа хватает общих знаний или предыдущих результатов "
    "инструментов — просто отвечай текстом, не вызывай инструменты без нужды."
)

MAX_STEPS = 5  # предохранитель от бесконечного цикла (модель зациклилась на вызовах)


def run_agent_verbose(
    user_message: str,
    history: list[dict] = None,
    extra_instruction: str = None,
    verbose: bool = True,
) -> dict:
    """То же самое, что run_agent, но возвращает ещё и trace — какие
    инструменты вызывались и что они вернули. Нужен для orchestrator.py:
    критик должен видеть, на чём основан ответ, а не только сам ответ,
    иначе он проверяет "звучит ли складно", а не "правда ли это".

    history — предыдущие сообщения диалога ([{"role": "user"/"assistant",
    "content": ...}, ...], уже обрезанные memory.py). Мы намеренно НЕ храним
    в истории промежуточные tool-call сообщения прошлых запросов — только
    финальные пары вопрос/ответ. Это проще и меньше по объёму; агент в
    рамках ТЕКУЩЕГО вопроса всё равно может вызвать инструменты заново,
    если ему нужны свежие данные, а не то, что "помнит" из прошлого ответа.

    extra_instruction — необязательная добавка к system prompt. Используется
    orchestrator.py при повторной попытке: "вот что не так с предыдущим
    ответом, поправь" — без этого агент просто повторил бы тот же ответ.
    """
    client = Groq(api_key=config.GROQ_API_KEY)
    system = SYSTEM_PROMPT
    if extra_instruction:
        system += f"\n\nВАЖНО: предыдущий ответ был отклонён проверкой. {extra_instruction}"

    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})
    trace = []

    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {"answer": msg.content, "trace": trace}

        messages.append(msg)

        for call in msg.tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments)
            if verbose:
                print(f"  [шаг {step+1}] вызов инструмента: {fn_name}({args})")

            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn(**args) if fn else f"Неизвестный инструмент: {fn_name}"
            trace.append({"tool": fn_name, "args": args, "result": str(result)})

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return {"answer": "Достигнут лимит шагов — агент не смог прийти к финальному ответу.", "trace": trace}


def run_agent(user_message: str, history: list[dict] = None, verbose: bool = True) -> str:
    """Обёртка для обратной совместимости (api.py, CLI) — просто текст ответа,
    без trace. Используй run_agent_verbose напрямую, если нужен trace."""
    return run_agent_verbose(user_message, history=history, verbose=verbose)["answer"]


if __name__ == "__main__":
    q = input("Вопрос: ")
    print("\n" + run_agent(q))
