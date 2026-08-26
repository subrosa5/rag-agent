"use client";

import { useState, useRef, useEffect } from "react";

type Message = { role: "user" | "assistant"; content: string };

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send() {
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      const answer = res.ok
        ? data.answer
        : `Ошибка: ${data.error ?? "не удалось получить ответ"}`;
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Не удалось связаться с бэкендом." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto">
      <header className="px-6 py-4 border-b border-neutral-800">
        <h1 className="text-lg font-semibold">rag-agent</h1>
        <p className="text-sm text-neutral-400">
          Агент с инструментами: поиск по документам (RAG) + калькулятор.
          Модель сама решает, что вызвать.
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-neutral-500 text-sm space-y-2">
            <p>Попробуй, например:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Сколько времени занимает откат деплоя?</li>
              <li>Когда новый сотрудник получает доступ к продакшн-базе?</li>
              <li>Сколько будет 128 * 17 + 4?</li>
            </ul>
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2 whitespace-pre-wrap ${
                m.role === "user"
                  ? "bg-neutral-100 text-neutral-900"
                  : "bg-neutral-800 text-neutral-100"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl px-4 py-2 bg-neutral-800 text-neutral-400 text-sm">
              думает…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="px-6 py-4 border-t border-neutral-800 flex gap-2">
        <input
          className="flex-1 rounded-xl bg-neutral-800 px-4 py-2 outline-none focus:ring-1 focus:ring-neutral-500"
          placeholder="Спроси что-нибудь…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={loading}
        />
        <button
          onClick={send}
          disabled={loading}
          className="rounded-xl bg-neutral-100 text-neutral-900 px-4 py-2 font-medium disabled:opacity-50"
        >
          Отправить
        </button>
      </div>
    </div>
  );
}
