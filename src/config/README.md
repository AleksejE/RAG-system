# config

Настройки проекта.

| Файл | Назначение |
|---|---|
| `settings.py` | Класс `Settings` на основе pydantic-settings. Читает переменные из `.env`, проверяет их типы и создаёт объект `settings`, который импортируют остальные модули. Ключ `GROQ_API_KEY` хранится как `SecretStr`, поэтому не выводится в логах и при `print`. |

## Какие настройки есть

| Группа | Переменные |
|---|---|
| LLM | `GROQ_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE` |
| Эмбеддинги | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `EMBEDDING_DIM` |
| Qdrant | `QDRANT_URL`, `COLLECTION_NAME` |
| Данные | `SOURCE_URL` |
| RAG | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `PAGE_SIZE_CHARS`, `TOP_K_RESULTS` |

## Как добавить новую настройку

1. Добавить поле в класс `Settings` с типом и значением по умолчанию.
2. Добавить переменную в `.env.example` с комментарием.
