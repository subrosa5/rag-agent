// Прокси-роут: браузер стучится сюда, а не напрямую в Render.
//
// Почему прокси, а не прямой fetch с клиента на Render API:
// 1) RAG_AGENT_API_URL остаётся server-only переменной (без NEXT_PUBLIC_
//    префикса) — адрес бэкенда не попадает в JS-бандл, который видит браузер.
// 2) Запрос идёт сервер-сервер (Vercel function -> Render), поэтому CORS
//    вообще не участвует — это НЕ то же самое, что "CORS настроен
//    неправильно", просто в этой схеме браузер никогда не делает
//    cross-origin запрос напрямую. CORS-мидлварь в FastAPI (src/api.py)
//    остаётся как fallback для прямых вызовов API (curl, тестирование).
export async function POST(request: Request) {
  const apiUrl = process.env.RAG_AGENT_API_URL;
  if (!apiUrl) {
    return Response.json(
      { error: "RAG_AGENT_API_URL не задан на сервере фронта" },
      { status: 500 }
    );
  }

  const body = await request.json();

  const upstream = await fetch(`${apiUrl}/agent/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok) {
    return Response.json(
      { error: `Бэкенд ответил ${upstream.status}` },
      { status: 502 }
    );
  }

  const data = await upstream.json();
  return Response.json(data);
}
