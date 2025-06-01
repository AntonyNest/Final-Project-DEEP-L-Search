# FastAPI entry point

"""
FastAPI Application Entry Point - Головна точка входу в систему.

Цей модуль виступає як Application Controller в архітектурному патерні MVC.
Він відповідає за:

1. **Ініціалізацію додатку** - налаштування FastAPI з усіма потрібними компонентами
2. **Middleware Pipeline** - обробка запитів на різних рівнях (CORS, логування, помилки)
3. **Dependency Injection** - забезпечення залежностей для endpoints
4. **Application Lifecycle** - startup та shutdown events
5. **Route Registration** - підключення всіх API endpoints
6. **Error Handling** - централізована обробка помилок

Архітектурний підхід: Layered Architecture
- Presentation Layer (FastAPI routes)
- Application Layer (services coordination) 
- Domain Layer (business logic)
- Infrastructure Layer (databases, external APIs)
"""

import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware

# Наші внутрішні компоненти
from app.config import settings
from app.services.search_service import search_service
from app.models.schemas import ErrorResponse, ErrorDetail
from app.utils.logger import setup_logging
from app.api.endpoints import search, documents

# Налаштовуємо логування перед створенням додатку
setup_logging()
logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для вимірювання часу обробки запитів.
    
    Цей компонент критично важливий для моніторингу продуктивності
    ML систем, де час відгуку може сильно варіюватися залежно від
    розміру запиту та навантаження на модель.
    
    Архітектурний патерн: Decorator Pattern
    Обгортає обробку запиту додатковою функціональністю без зміни основної логіки.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Записуємо час початку обробки
        start_time = time.time()
        
        # Логуємо вхідний запит для debugging
        logger.info(
            f"🔄 Processing request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Виконуємо основну обробку запиту
            response = await call_next(request)
            
            # Обчислюємо час обробки
            processing_time = time.time() - start_time
            
            # Додаємо header з часом обробки для клієнта
            response.headers["X-Process-Time"] = str(round(processing_time * 1000, 2))
            
            # Логуємо успішну відповідь
            logger.info(
                f"✅ Request completed: {request.method} {request.url.path} "
                f"Status: {response.status_code} Time: {processing_time:.3f}s"
            )
            
            return response
            
        except Exception as exc:
            # Логуємо помилку з повною трасою
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Request failed: {request.method} {request.url.path} "
                f"Time: {processing_time:.3f}s Error: {str(exc)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # Перекидаємо виключення для обробки в exception handlers
            raise exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager для додатку.
    
    Цей async context manager керує життєвим циклом додатку:
    - startup: ініціалізація сервісів, підключення до БД, завантаження ML моделей
    - shutdown: graceful зупинка, закриття з'єднань, cleanup ресурсів
    
    Архітектурний патерн: Resource Acquisition Is Initialization (RAII)
    Забезпечуємо правильне управління ресурсами в async середовищі.
    """
    # === STARTUP PHASE ===
    logger.info("🚀 Starting Document Search Service...")
    
    try:
        # Перевірка конфігурації
        logger.info("🔧 Validating configuration...")
        if not settings.documents_path_obj.exists():
            logger.warning(f"Documents path does not exist: {settings.documents_path}")
        
        # Ініціалізація пошукової системи
        logger.info("🧠 Initializing AI components...")
        initialization_success = await search_service.initialize_system()
        
        if not initialization_success:
            logger.error("❌ Failed to initialize search system")
            # В production тут можна було б припинити запуск
            # raise RuntimeError("Search system initialization failed")
        else:
            logger.info("✅ Search system initialized successfully")
        
        # Отримуємо статистику системи для початкового стану
        stats = search_service.get_document_stats()
        logger.info(f"📊 System stats: {stats}")
        
        logger.info("🎉 Application startup completed successfully!")
        
        # Yield означає що додаток готовий приймати запити
        yield
        
    except Exception as e:
        logger.error(f"💥 Startup failed: {str(e)}")
        raise
    
    # === SHUTDOWN PHASE ===
    logger.info("🛑 Shutting down Document Search Service...")
    
    try:
        # Очищення кешів для звільнення пам'яті
        logger.info("🧹 Cleaning up caches...")
        cache_result = search_service.clear_cache()
        logger.info(f"Cache cleanup result: {cache_result}")
        
        # Можна додати збереження статистики, закриття з'єднань і т.д.
        logger.info("💾 Saving final statistics...")
        
        logger.info("👋 Application shutdown completed")
        
    except Exception as e:
        logger.error(f"⚠️ Error during shutdown: {str(e)}")


def create_application() -> FastAPI:
    """
    Factory function для створення FastAPI додатку.
    
    Цей підхід дозволяє легко створювати різні конфігурації додатку
    для development, testing та production середовищ.
    
    Архітектурний патерн: Factory Pattern
    Інкапсулює складну логіку створення об'єкту в одній функції.
    """
    # Створюємо базовий FastAPI додаток з lifecycle management
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        🔍 **Сервіс семантичного пошуку документів**
        
        Система для інтелектуального пошуку та аналізу документів з використанням 
        штучного інтелекту та векторних баз даних.
        
        **Основні можливості:**
        - 🧠 Семантичний пошук з використанням ембедингів
        - 📄 Підтримка .docx, .doc, .pdf форматів
        - ⚡ Високошвидкісний векторний пошук через Qdrant
        - 🎯 Фільтрація за метаданими документів
        - 📊 Детальна аналітика та статистика
        
        **Технології:**
        - FastAPI + Pydantic для API
        - Sentence Transformers для ембедингів  
        - Qdrant як векторна база даних
        - Docker для контейнеризації
        """,
        lifespan=lifespan,  # Підключаємо lifecycle management
        docs_url="/docs",   # Swagger UI доступна на /docs
        redoc_url="/redoc"  # ReDoc доступна на /redoc
    )
    
    return app


def configure_cors(app: FastAPI) -> None:
    """
    Налаштування CORS (Cross-Origin Resource Sharing).
    
    CORS критично важливий для web додатків, які звертаються до API
    з браузера. Налаштовуємо безпечні але гнучкі правила.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # Дозволені домени
        allow_credentials=True,               # Дозволити cookies/auth headers
        allow_methods=["GET", "POST", "PUT", "DELETE"],  # HTTP методи
        allow_headers=["*"],                  # Дозволені headers
        expose_headers=["X-Process-Time"]     # Headers видимі клієнту
    )
    logger.info(f"✅ CORS configured for origins: {settings.cors_origins}")


def configure_security(app: FastAPI) -> None:
    """
    Налаштування базової безпеки додатку.
    
    Додаємо middleware для захисту від основних атак:
    - Host header injection
    - Timing attacks через consistent response times
    """
    # Захист від Host header attacks
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # В production варто обмежити конкретними доменами
    )
    
    # Додаємо кастомний timing middleware
    app.add_middleware(TimingMiddleware)
    
    logger.info("🔒 Security middleware configured")


def configure_exception_handlers(app: FastAPI) -> None:
    """
    Централізована обробка помилок.
    
    Цей компонент забезпечує консистентний формат відповідей
    при помилках та захищає від витоку внутрішньої інформації.
    
    Архітектурний патерн: Chain of Responsibility
    Різні типи помилок обробляються різними handlers.
    """
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Обробка стандартних HTTP помилок."""
        logger.warning(
            f"HTTP Exception: {exc.status_code} {exc.detail} "
            f"Path: {request.url.path}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                success=False,
                message="HTTP Error occurred",
                error=ErrorDetail(
                    error_code=f"HTTP_{exc.status_code}",
                    error_message=str(exc.detail),
                    error_details={"status_code": exc.status_code}
                )
            ).dict()
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Обробка помилок валідації даних."""
        logger.error(f"Validation Error: {str(exc)} Path: {request.url.path}")
        
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                success=False,
                message="Validation error",
                error=ErrorDetail(
                    error_code="VALIDATION_ERROR",
                    error_message=str(exc),
                    error_details={"path": str(request.url.path)}
                )
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Обробка всіх неочікуваних помилок."""
        # Логуємо повну трасу помилки для debugging
        logger.error(
            f"Unhandled Exception: {type(exc).__name__}: {str(exc)} "
            f"Path: {request.url.path}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                message="Internal server error occurred",
                error=ErrorDetail(
                    error_code="INTERNAL_SERVER_ERROR",
                    error_message="An unexpected error occurred. Please try again later.",
                    error_details={
                        "error_type": type(exc).__name__,
                        "path": str(request.url.path),
                        # В production не включаємо детальну інформацію про помилки
                        "debug_info": str(exc) if settings.log_level == "DEBUG" else None
                    }
                )
            ).dict()
        )
    
    logger.info("🚨 Exception handlers configured")


def register_routes(app: FastAPI) -> None:
    """
    Реєстрація всіх API routes.
    
    Організовуємо endpoints в логічні групи для кращої структури API.
    Кожна група endpoints знаходиться в окремому модулі.
    """
    # API версіонування - всі routes під /api/v1/
    API_V1_PREFIX = "/api/v1"
    
    # Endpoints для пошуку документів
    app.include_router(
        search.router,
        prefix=f"{API_V1_PREFIX}/search",
        tags=["Search"],  # Для групування в Swagger UI
        responses={
            404: {"description": "Not found"},
            500: {"description": "Internal server error"}
        }
    )
    
    # Endpoints для управління документами
    app.include_router(
        documents.router,
        prefix=f"{API_V1_PREFIX}/documents", 
        tags=["Documents"],
        responses={
            404: {"description": "Not found"},
            500: {"description": "Internal server error"}
        }
    )
    
    logger.info("🛣️ API routes registered")


def customize_openapi_schema(app: FastAPI) -> None:
    """
    Кастомізація OpenAPI схеми для кращої документації.
    
    Додаємо детальну інформацію для автогенерації якісної
    API документації в Swagger UI та ReDoc.
    """
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=settings.app_name,
            version=settings.app_version,
            description=app.description,
            routes=app.routes,
        )
        
        # Додаємо кастомну інформацію
        openapi_schema["info"].update({
            "contact": {
                "name": "Document Search API Support",
                "email": "support@example.com"  # Змініть на реальний email
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        })
        
        # Додаємо загальні responses для всіх endpoints
        openapi_schema["components"]["responses"] = {
            "ValidationError": {
                "description": "Validation error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            "InternalError": {
                "description": "Internal server error", 
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    logger.info("📚 OpenAPI schema customized")


# === Створення та налаштування додатку ===

# Створюємо головний екземпляр додатку
app = create_application()

# Налаштовуємо всі компоненти
configure_cors(app)
configure_security(app)
configure_exception_handlers(app)
register_routes(app)
customize_openapi_schema(app)


# === Health Check Endpoint ===

@app.get("/health", tags=["System"])
async def health_check():
    """
    Базовий health check endpoint.
    
    Цей endpoint критично важливий для:
    - Load balancers для перевірки живості сервісу
    - Monitoring систем для алертів
    - Kubernetes health probes
    - CI/CD pipelines для перевірки deployment
    """
    try:
        # Перевіряємо стан основних компонентів
        search_stats = search_service.get_document_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.app_version,
            "components": {
                "search_service": "ok" if search_stats else "error",
                "vector_database": search_stats.get("system_health", {}).get("vector_db_healthy", False),
                "ml_model": search_stats.get("system_health", {}).get("model_loaded", False)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.get("/", tags=["System"])
async def root():
    """
    Кореневий endpoint з базовою інформацією про API.
    
    Корисно для швидкої перевірки що сервіс працює
    та отримання посилань на документацію.
    """
    return {
        "message": f"🔍 {settings.app_name} v{settings.app_version}",
        "description": "Semantic Document Search API",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "health_check": "/health",
        "api_prefix": "/api/v1"
    }


# === Функція для запуску в development режимі ===

def run_development_server():
    """
    Запуск development сервера.
    
    Ця функція використовується тільки для локальної розробки.
    В production використовуємо gunicorn або інший WSGI сервер.
    """
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,      # Автоматичний перезапуск при зміні коду
        log_level=settings.log_level.lower(),
        access_log=True   # Логування всіх HTTP запитів
    )


# Точка входу для direct execution
if __name__ == "__main__":
    logger.info("🚀 Starting development server...")
    run_development_server()