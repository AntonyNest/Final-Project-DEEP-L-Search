# Final-Project-DEEP-L-Search

## Description
The goal of the project is to implement a contextual search service. The service should be able to search for relevant documents based on the user's query. The service should be able to handle a large number of documents and should be able to handle a large number of queries.

An intelligent document search system that uses modern machine learning technologies for semantic analysis and content search in .docx, .doc, .pdf format documents.

🧠 Semantic Search: Understands context and synonyms, not just exact matches
🇺🇦 Ukrainian Language Support: Optimized for working with Ukrainian texts
📄 Multiple formats: .docx, .doc, .pdf documents
⚡ Quick Search: Vector search using Qdrant
🔧 REST API: Ready endpoints for integration
🐳 Docker Ready: Full containerization for easy deployment
📊 Analytics: Detailed statistics and monitoring

## System Architecture
Our system will consist of four main components:
Document Processor - processing .doc/.docx files and text extraction
Embedding Service - generation of vector representations of text
Vector Database (Qdrant) - storage of embeddings and metadata
Search API - REST API for searching and interacting with the system

graph TD
    A[Documents] --> B[Document Processor]
    B --> C[Text Extraction]
    C --> D[Text Chunking]
    D --> E[Embedding Service]
    E --> F[Vector Database<br/>Qdrant]
    
    G[User Query] --> H[Search API]
    H --> E
    E --> I[Vector Search]
    F --> I
    I --> J[Post-processing]
    J --> K[Search Results]
    
    style E fill:#e1f5fe
    style F fill:#f3e5f5
    style H fill:#e8f5e8

## Prerequisites

Python 3.11+

Docker and Docker Compose
Minimum 4GB RAM
GPU (optional, for acceleration)

1. Cloning the repository
git clone https://github.com/AntonyNest/Final-Project-DEEP-L-Search.git
cd DOCUMENTS_PATH-search-service

2. Environment setup
### Копіюємо приклад конфігурації
cp .env.example .env

### Редагуємо конфігурацію під ваші потреби
nano .env

DOCUMENTS_PATH="C:\\Users\\YourName\\Documents\\MyDocs"

3. Run through Docker Compose
### Запуск всіх сервісів
docker-compose up -d

### Перевірка статусу
docker-compose ps

### Перегляд логів
docker-compose logs -f document-search-api

4. Local launch
### Створення віртуального середовища
python -m venv venv
source venv/bin/activate  # Linux/Mac or venv\Scripts\activate     # Windows

### Встановлення залежностей
pip install -r requirements.txt

### Запуск Qdrant (окремо)
docker run -p 6333:6333 qdrant/qdrant

### Запуск додатку
python -m uvicorn app.main:app --reload

5. Checking the work
Відкрийте у браузері:

API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health
Qdrant Dashboard: http://localhost:6333/dashboard

### Використання API
#### Indexing документів
Спочатку потрібно проіндексувати ваші документи:

curl -X POST "http://localhost:8000/api/v1/documents/index" \
  -H "Content-Type: application/json" \
  -d '{
    "custom_path": "/path/to/your/documents",
    "force_reindex": false
  }'

#### Семантичний пошук

curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "інформаційне забезпечення системи управління",
    "limit": 10,
    "score_threshold": 0.7,
    "include_stats": true
  }'

#### Швидкий пошук (GET)

curl "http://localhost:8000/api/v1/search/quick?q=архітектура%20системи&limit=5"

#### Статистика системи

curl "http://localhost:8000/api/v1/search/stats"

### Configuration
Основні параметри

DOCUMENTS_PATH- Шлях до документів (обов'язково)
QDRANT_HOST- Хост Qdrant (localhost)
EMBEDDING_MODEL - ML модель (paraphrase-multilingual-MiniLM-L12-v2)
MAX_CHUNK_SIZE - Розмір фрагменту (1000) 
SIMILARITY_THRESHOLD - Поріг схожості (0.5)

### ML модель налаштування
Для кращої роботи з українською мовою:

#### Балансована модель (рекомендовано)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

#### Вища точність (повільніше)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

#### Швидша робота (менша точність)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

## Тестування
Запуск тестів

### Всі тести
pytest

### З покриттям коду
pytest --cov=app tests/

### Тільки швидкі тести
pytest -m "not slow"

### Тестування конкретного компонента
pytest tests/test_search_api.py -v

### Тестування API
#### Тестування індексації
python scripts/test_search.py

#### Bulk тестування
python scripts/test_bulk_operations.py

# Моніторинг та логування
Логи зберігаються в logs/document_search.log (JSON формат):
### Перегляд логів в реальному часі
tail -f logs/document_search.log | jq .

### Пошук помилок
grep "ERROR" logs/document_search.log | jq .

### Метрики
Система надає детальні метрики через API:

#### Статистика системи
curl http://localhost:8000/api/v1/documents/stats/detailed

#### Health check
curl http://localhost:8000/health


🔍 Приклади використання
1. Пошук технічної документації
pythonimport requests

response = requests.post("http://localhost:8000/api/v1/search/semantic", json={
    "query": "технічні вимоги до архітектури платформи",
    "limit": 5,
    "score_threshold": 0.8,
    "file_types": ["docx", "pdf"]
})

results = response.json()
for result in results["results"]:
    print(f"📄 {result['source_file']}")
    print(f"🎯 Score: {result['score']:.3f}")
    print(f"📝 {result['text'][:200]}...")
    print("-" * 50)

2. Аналіз якості пошуку
python# Аналіз запиту перед пошуком
response = requests.post("http://localhost:8000/api/v1/search/analyze", json={
    "query": "інформаційна безпека"
})

analysis = response.json()
print(f"Складність запиту: {analysis['estimated_complexity']}")
print(f"Рекомендації: {analysis['recommendations']}")

3. Batch обробка
python# Переіндексація всіх документів
response = requests.post("http://localhost:8000/api/v1/documents/index", json={
    "force_reindex": True,
    "file_types_filter": ["docx", "pdf"]
})

stats = response.json()["stats"]
print(f"Оброблено: {stats['chunks_processed']} фрагментів")
print(f"Проіндексовано: {stats['chunks_indexed']} фрагментів")
print(f"Час обробки: {stats['total_time_s']:.2f} секунд")
🛠️ Розробка
Налаштування dev середовища
bash# Клонування та встановлення
git clone https://github.com/your-org/document-search-service.git
cd document-search-service

### Dev залежності
pip install -r requirements.txt
pip install pytest black flake8 mypy

### Pre-commit hooks
pre-commit install


# Структура проекту
document-search-service/
├── app/
│   ├── api/                 # REST API endpoints
│   ├── services/            # Бізнес логіка
│   ├── models/              # Дата моделі
│   ├── utils/               # Утиліти
│   ├── config.py            # Конфігурація
│   └── main.py              # FastAPI додаток
├── tests/                   # Тести
├── scripts/                 # Допоміжні скрипти
├── docs/                    # Документація
├── docker-compose.yml       # Docker Compose
├── Dockerfile               # Docker образ
└── requirements.txt         # Python залежності


## Code Style
Проект використовує:

Black для форматування коду
Flake8 для лінтінгу
MyPy для перевірки типів
Pytest для тестування

# Kubernetes
Docker Production
## Build production образу
docker build --target production -t document-search:latest .

## Запуск з production конфігурацією
docker run -d \
  --name document-search \
  -p 8000:8000 \
  -v /path/to/documents:/app/documents:ro \
  -v /path/to/logs:/app/logs \
  --env-file .env.production \
  document-search:latest

## k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-search
  template:
    metadata:
      labels:
        app: document-search
    spec:
      containers:
      - name: document-search
        image: document-search:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: document-search-config
        volumeMounts:
        - name: documents
          mountPath: /app/documents
          readOnly: true


👥 Автори

AntonyNest - Початкова робота - YourOrg