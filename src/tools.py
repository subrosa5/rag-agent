"""tools.py — инструменты, которые агент может вызывать.

Два инструмента нарочно очень разной природы — это делает "агентность"
видимой: модель должна САМА понять, какой инструмент нужен под конкретный
вопрос, а не просто всегда дёргать один и тот же.

  search_documents — RAG как инструмент: агент лезет в базу знаний,
                      когда вопрос требует фактов из документов.
  calculate        — обычная функция: агент считает, когда вопрос про число,
                      а не про знание из документов.

Каждый инструмент описан в двух местах:
  1) TOOLS_SCHEMA — JSON-описание для модели (что это, какие параметры) —
     ничего "не выполняет", это просто текстовая инструкция для LLM.
  2) TOOL_FUNCTIONS — реальная python-функция, которую МЫ вызываем, когда
     модель попросила именно этот tool_call. Модель никогда не выполняет
     код сама — она только просит, выполнение — всегда на нашей стороне.
"""
import ast
import operator

from .retrieval import hybrid_retrieve


def search_documents(query: str) -> str:
    """Ищет в базе документов и возвращает найденные куски с источниками.
    Не генерирует ответ сама — этим займётся LLM в основном цикле агента,
    у неё будет и это, и вся остальная история диалога."""
    chunks = hybrid_retrieve(query)
    if not chunks:
        return "Ничего не найдено в базе документов."
    return "\n\n".join(
        f"[источник: {c['source']}] {c['content']}" for c in chunks
    )


# --- безопасный калькулятор ---
# Мы намеренно НЕ используем eval(expression) — LLM генерирует строку,
# которую мы выполняем как код, а это классическая дыра (модель могла
# "нафантазировать" что угодно, включая __import__('os').system(...)).
# Вместо этого разбираем выражение через ast и разрешаем только арифметику.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Недопустимое выражение: {ast.dump(node)}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Найти релевантную информацию в базе документов пользователя. Используй, когда вопрос требует конкретных фактов из загруженных документов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос — переформулированный вопрос пользователя"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Посчитать арифметическое выражение (+, -, *, /, **). Используй для любой математики.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Например: '12 * (3 + 4)'"}
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "calculate": calculate,
}
