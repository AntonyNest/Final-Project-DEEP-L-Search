# app/__init__.py
"""
Document Search Service - AI-powered semantic document search system.

Цей пакет містить всі компоненти для інтелектуального пошуку документів:
- REST API на базі FastAPI
- ML pipeline для обробки документів
- Векторна база даних для семантичного пошуку
- Система управління та моніторингу

Основні компоненти:
- api: REST API endpoints та залежності
- services: Бізнес-логіка та ML сервіси  
- models: Дата моделі та схеми
- utils: Допоміжні утиліти та інфраструктура

Версія: 1.0.0
Автор: ML Engineering Team
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__email__ = "ml-team@example.com"

# Експортуємо основні компоненти для зручності імпорту
from app.main import app
from app.config import settings

__all__ = ["app", "settings", "__version__"]

# ================================================================

# app/api/__init__.py
"""
API Layer - REST API endpoints та їх залежності.

Цей пакет містить всі компоненти для HTTP API:
- endpoints: Конкретні API routes
- dependencies: Dependency injection система
- middleware: HTTP middleware компоненти

Архітектурний підхід: Layered Architecture
API layer відокремлений від бізнес-логіки та може бути легко замінений.
"""

from app.api.endpoints import search, documents

__all__ = ["search", "documents"]

# ================================================================

# app/api/endpoints/__init__.py
"""
API Endpoints - Конкретні HTTP endpoints для різних функцій.

Endpoints організовані за функціональними доменами:
- search: Пошукові операції
- documents: Управління документами та індексацією

Кожен модуль містить router який підключається до головного FastAPI app.
"""

from app.api.endpoints import search, documents

# Експортуємо routers для підключення в main app
search_router = search.router
documents_router = documents.router

__all__ = ["search_router", "documents_router", "search", "documents"]

# ================================================================

# app/models/__init__.py  
"""
Data Models - Моделі даних для API та внутрішньої логіки.

Цей пакет містить два типи моделей:
- schemas: Pydantic моделі для API (валідація, серіалізація)
- document: Domain моделі для внутрішньої бізнес-логіки

Розділення дозволяє:
- Незалежну еволюцію API та domain logic
- Чітке розмежування зовнішніх контрактів та внутрішньої реалізації
- Легше тестування та підтримку
"""

from app.models import schemas, document

# Експортуємо основні схеми для зручності
from app.models.schemas import (
    SearchRequest, SearchResponse, SearchResultItem,
    IndexingRequest, IndexingResponse,
    BaseResponse, SystemStatsResponse
)

from app.models.document import (
    Document, DocumentChunk, DocumentCollection,
    ProcessingStatus
)

__all__ = [
    # Modules
    "schemas", "document",
    
    # API Schemas
    "SearchRequest", "SearchResponse", "SearchResultItem",
    "IndexingRequest", "IndexingResponse",
    "BaseResponse", "SystemStatsResponse",
    
    # Domain Models  
    "Document", "DocumentChunk", "DocumentCollection",
    "ProcessingStatus"
]

# ================================================================

# app/services/__init__.py
"""
Services Layer - Бізнес-логіка та ML сервіси.

Цей пакет містить всі сервіси для обробки документів та пошуку:
- document_processor: Екстракція та обробка тексту з документів
- embedding_service: Генерація векторних представлень тексту
- vector_store: Робота з векторною базою даних
- search_service: Оркестрація пошукових операцій

Архітектурний підхід: Service Layer Pattern
Сервіси інкапсулюють бізнес-логіку та надають чистий API для використання.
"""

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store  
from app.services.search_service import search_service

# Експортуємо основні сервіси як singleton instances
__all__ = [
    "DocumentProcessor",
    "embedding_service", 
    "vector_store",
    "search_service"
]

# ================================================================

# app/utils/__init__.py
"""
Utilities - Допоміжні утиліти та інфраструктурні компоненти.

Цей пакет містить переспільні утиліти:
- logger: Система логування
- exceptions: Кастомні виключення
- helpers: Різні допоміжні функції

Ці компоненти використовуються скрізь в додатку для забезпечення
консистентності та зменшення дублювання коду.
"""

from app.utils.logger import setup_logging, get_ml_logger
from app.utils.exceptions import (
    DocumentSearchException,
    MLServiceError,
    DocumentProcessingError, 
    ValidationError
)

# Автоматично налаштовуємо логування при імпорті пакету
# setup_logging()  # Коментуємо тому що викликається в main.py

__all__ = [
    # Logging
    "setup_logging", "get_ml_logger",
    
    # Exceptions
    "DocumentSearchException",
    "MLServiceError", 
    "DocumentProcessingError",
    "ValidationError"
]