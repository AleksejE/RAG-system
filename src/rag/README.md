# rag

**Модуль RAG-chain.** Ищет фрагменты Конституции в Qdrant и генерирует ответ через LLM в Groq.

| Файл | Назначение |
|---|---|
| `prompts.py` | Системный промпт и шаблон `RAG_PROMPT`: отвечать только по контексту, ссылаться на статьи, отвечать на языке вопроса. В шаблоне есть необязательный `chat_history` для модуля памяти. |
| `chain.py` | Сборка цепочки на LCEL: retriever → контекст → промпт → `ChatGroq` → ответ и источники. |

## Как устроена цепочка

```
{"question"} → retriever (Qdrant, top-k) → format_docs → RAG_PROMPT → ChatGroq → StrOutputParser
                                     └──────────────→ sources (Header, article, page_number)
```

## Функции

- `build_rag_chain(k=None)` возвращает `Runnable`.
  - Вход: `{"question": str, "chat_history": list[BaseMessage] (необязательно)}`.
  - Выход: `{"answer": str, "sources": list[dict], "docs": list[Document]}`.
- `ask(question, k=None)` задаёт один вопрос без истории.
- `get_retriever(k=None)` возвращает retriever поверх коллекции Qdrant. Его же использует `analytics/evaluate_retrieval.py`.
- `get_llm()` возвращает `ChatGroq` с таймаутом и повторами (в том числе при rate limit). Если `GROQ_API_KEY` не задан, выбрасывает `ValueError`.

## Для модуля памяти

Историю передавайте в ключе `chat_history`, например через `RunnableWithMessageHistory`
с `input_messages_key="question"` и `history_messages_key="chat_history"`.
В выходе ответ лежит в ключе `answer` (`output_messages_key="answer"`).

## Параметры (из `.env`)

- `GROQ_API_KEY`: ключ Groq, обязателен.
- `LLM_MODEL` (по умолчанию `openai/gpt-oss-120b`), `LLM_TEMPERATURE` (0.1).
- `LLM_TIMEOUT_SEC` (60) и `LLM_MAX_RETRIES` (3).
- `TOP_K_RESULTS` (5): сколько чанков передаётся в контекст.

## Запуск

```bash
uv run python -m src.main ask "What is the term of office of the President?"
uv run python -m src.main ask "На какой срок избирается Президент?"
```
