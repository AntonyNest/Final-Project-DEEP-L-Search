# Dockerfile для Document Search Service
# 
# Цей Dockerfile створює production-ready контейнер для нашого
# AI-powered document search service з оптимізаціями для:
# 1. Швидкого запуску (multi-stage build)
# 2. Мінімального розміру (Alpine base)
# 3. Безпеки (non-root user)
# 4. Кешування залежностей (оптимізація layers)

# ================================================================
# Stage 1: Базовий образ з Python та системними залежностями
# ================================================================

FROM python:3.11-slim as base

# Метадані образу
LABEL maintainer="ML Engineering Team <ml-team@example.com>"
LABEL description="AI-powered semantic document search service"
LABEL version="1.0.0"

# Встановлюємо змінні середовища для Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1

# Встановлюємо системні залежності
# Розділяємо на кілька RUN команд для кращого кешування
RUN apt-get update && apt-get install -y \
    # Загальні системні утиліти
    curl \
    wget \
    unzip \
    # Бібліотеки для обробки документів
    libxml2-dev \
    libxslt1-dev \
    # Залежності для PDF обробки
    poppler-utils \
    # Бібліотеки для ML (якщо потрібні native extensions)
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# Stage 2: Встановлення Python залежностей
# ================================================================

FROM base as dependencies

# Копіюємо файли залежностей
COPY requirements.txt /tmp/requirements.txt

# Встановлюємо Python залежності
# Використовуємо pip install з оптимізаціями
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Попередньо завантажуємо ML модель для швидшого запуску
# Це опціонально - можна також робити lazy loading
RUN python -c "
from sentence_transformers import SentenceTransformer;
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2');
print('Model downloaded successfully')
"

# ================================================================
# Stage 3: Production образ
# ================================================================

FROM base as production

# Створюємо користувача для безпеки (non-root)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Створюємо директорії для додатку
RUN mkdir -p /app/logs /app/cache /app/documents && \
    chown -R appuser:appuser /app

# Копіюємо встановлені залежності з попереднього stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо код додатку
# Спочатку копіюємо структуру, потім код для кращого кешування
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY *.py /app/

# Встановлюємо права доступу
RUN chown -R appuser:appuser /app && \
    chmod +x /app/scripts/*.py

# Переключаємося на non-root користувача
USER appuser

# Встановлюємо змінні середовища для додатку
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    DOCUMENTS_PATH=/app/documents \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333

# Експонуємо порт
EXPOSE 8000

# Health check для моніторингу
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Команда запуску за замовчуванням
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ================================================================
# Stage 4: Development образ (опціонально)
# ================================================================

FROM production as development

# Переключаємося назад на root для встановлення dev залежностей
USER root

# Встановлюємо додаткові dev інструменти
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    jupyter

# Встановлюємо git для development
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Переключаємося назад на appuser
USER appuser

# Змінюємо команду для development (з auto-reload)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================================================
# Інструкції для збірки:
# 
# Production build:
# docker build --target production -t document-search:latest .
# 
# Development build:
# docker build --target development -t document-search:dev .
# 
# Multi-platform build (для ARM/Intel):
# docker buildx build --platform linux/amd64,linux/arm64 -t document-search:latest .
# ================================================================