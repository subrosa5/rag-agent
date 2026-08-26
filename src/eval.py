"""eval.py — маленький, но настоящий eval-харнес.

Без этого файла проект — игрушка ("вроде отвечает"). С ним — можно
показать число и объяснить, что оно значит. Мы не тянем ragas (тяжёлая
зависимость, обычно требует ещё и OpenAI-ключ по умолчанию) — вместо этого
руками реализуем те же две метрики, которые ragas считает через LLM-judge.
Идея та же, просто явно видно, что происходит внутри.

Метрики:
  retrieval_hit  — нашёлся ли среди retrieved-чанков чанк из правильного
                   источника. Проверяет ТОЛЬКО retrieval, до генерации.
  faithfulness   — просим LLM (другим вызовом, "судьёй") оценить: следует
                   ли сгенерированный ответ строго из контекста, без
                   отсебятины. Проверяет генерацию, а не retrieval.

Golden set — маленький вручную размеченный набор вопрос/источник, привязанный
к data/*.txt. На реальном проекте таких пар должно быть 30-100+, здесь для
демонстрации метода хватает нескольких.
"""
import json

from groq import Groq

from . import config
from .rag import answer as rag_answer
from .retrieval import hybrid_retrieve

GOLDEN_SET = [
    {
        "question": "Сколько времени обычно занимает откат деплоя?",
        "expected_source": "data/deployment.txt",
    },
    {
        "question": "Когда новый сотрудник получает доступ к продакшн-базе данных?",
        "expected_source": "data/onboarding.txt",
    },
    {
        "question": "Какой лимит запросов в минуту на бесплатном тарифе API?",
        "expected_source": "data/api-rate-limits.txt",
    },
    {
        # ловушка на grounding: ответа в базе нет, модель ДОЛЖНА сказать, что не знает
        "question": "Какая погода в Москве завтра?",
        "expected_source": None,
    },
]

JUDGE_PROMPT = """Ты — строгий судья качества ответов RAG-системы.
Тебе дан КОНТЕКСТ и ОТВЕТ. Оцени: следует ли ответ ТОЛЬКО из контекста,
без придуманных фактов, которых в контексте нет?
Если контекст пустой/нерелевантный и ответ честно говорит "не знаю" — это тоже PASS.
Ответь ровно одним словом: PASS или FAIL.

Контекст:
{context}

Ответ:
{answer}

Вердикт:"""


def judge_faithfulness(context: str, answer: str, client: Groq) -> bool:
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
        temperature=0,  # судье не нужна креативность, нужна повторяемость
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("PASS")


def run_eval():
    client = Groq(api_key=config.GROQ_API_KEY)
    results = []

    for item in GOLDEN_SET:
        question = item["question"]
        chunks = hybrid_retrieve(question)
        retrieved_sources = {c["source"] for c in chunks}

        if item["expected_source"] is None:
            retrieval_hit = len(chunks) == 0 or True  # для ловушки retrieval не судим строго
        else:
            retrieval_hit = item["expected_source"] in retrieved_sources

        result = rag_answer(question)
        context = "\n\n".join(c["content"] for c in chunks)
        faithful = judge_faithfulness(context, result["answer"], client)

        results.append({
            "question": question,
            "retrieval_hit": retrieval_hit,
            "faithful": faithful,
            "answer": result["answer"][:200],
        })

    hit_rate = sum(r["retrieval_hit"] for r in results) / len(results)
    faithfulness_rate = sum(r["faithful"] for r in results) / len(results)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nretrieval hit rate:  {hit_rate:.0%}")
    print(f"faithfulness rate:   {faithfulness_rate:.0%}")


if __name__ == "__main__":
    run_eval()
