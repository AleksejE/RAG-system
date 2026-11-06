# rag-system-constitution

RAG-система (Retrieval-Augmented Generation), которая отвечает на вопросы о
**Конституции Российской Федерации**. Текст Конституции скачивается с сайта
[constitution.garant.ru](https://constitution.garant.ru/english/), режется на
фрагменты по статьям, переводится в эмбеддинги моделью `BAAI/bge-m3` и
сохраняется в векторной базе Qdrant. На вопрос пользователя система находит
самые похожие фрагменты и передаёт их языковой модели через Groq, которая
отвечает со ссылкой на раздел и страницу.

## Технологии

| Что | Чем |
|---|---|
| Язык | Python 3.11+ |
| Менеджер зависимостей | [uv](https://docs.astral.sh/uv/) |
| Настройки | pydantic-settings (`.env`) |
| Парсинг | requests + BeautifulSoup |
| Оркестрация RAG | LangChain |
| Эмбеддинги | `BAAI/bge-m3` (HuggingFace, локально) |
| Векторная БД | Qdrant (в Docker) |
| LLM | `openai/gpt-oss-120b` через Groq API |
| Линтер и форматтер | ruff |

## Структура репозитория

```
rag-system-constitution/
├── src/                  # исходный код приложения
│   ├── config/           # настройки проекта (чтение .env)
│   ├── parsing/          # скачивание и разбор текста Конституции
│   ├── chunking/         # разбиение текста на чанки с метаданными
│   ├── embeddings/       # модель эмбеддингов bge-m3
│   ├── vector_store/     # модуль БД: работа с Qdrant
│   └── main.py           # точка входа: index / search
├── experiments/          # эксперименты с параметрами RAG
├── analytics/            # оценка качества ответов
├── docker-compose.yml    # запуск Qdrant
├── .env.example          # пример переменных окружения
├── pyproject.toml        # зависимости и настройки ruff
└── README.md
```

В каждой папке есть свой `README.md` с описанием её файлов.

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/KimVladimirV/rag-system-constitution.git
cd rag-system-constitution

# 2. Установить зависимости (uv сам создаст окружение .venv)
uv sync

# 3. Создать файл с настройками (GROQ_API_KEY нужен только модулю RAG)
cp .env.example .env

# 4. Запустить Qdrant
docker compose up -d

# 5. Скачать Конституцию и построить векторный индекс
uv run python -m src.main index

# 6. Проверить, что поиск работает
uv run python -m src.main search "term of office of the President"
```

При первом запуске модель `bge-m3` (около 2 ГБ) скачается с HuggingFace.

## Проверка кода

```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint + auto-fix
uv run ruff format .         # format
```

## Правила работы с репозиторием

1. Не пушим напрямую в `main`: каждая задача делается в отдельной ветке
   (`feature/...`, `fix/...`) и вливается через pull request.
2. Перед коммитом запускаем `ruff check` и `ruff format`.
3. Каждый `.py`-файл начинается с docstring о назначении файла, у каждой
   крупной функции есть docstring в формате Google.
4. При добавлении новой папки в `src/` создаём в ней `README.md`.
5. Ключи API храним только в `.env`, который не попадает в git.
6. Папки `.venv/` и `qdrant_storage/` в репозиторий не коммитим.
