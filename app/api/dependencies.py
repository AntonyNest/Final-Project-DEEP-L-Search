# FastAPI dependencies

"""
API Dependencies для Document Search Service.

Dependency Injection є ключовим архітектурним патерном у FastAPI.
Цей модуль визначає всі залежності, які можуть бути автоматично
ін'єктовані в API endpoints.

Переваги Dependency Injection:
1. **Testability** - легко mock-ати залежності для тестів
2. **Loose Coupling** - endpoints не залежать від конкретних реалізацій
3. **Reusability** - одні й ті ж залежності використовуються скрізь
4. **Lifecycle Management** - автоматичне управління життєвим циклом об'єктів

Архітектурний підхід: Service Locator Pattern
FastAPI виступає як service locator, який автоматично резолвить
та ін'єктує потрібні залежності в endpoints.
"""

import logging
import time
from typing import Optional, Dict, Any, Annotated
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.services.search_service import search_service
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.utils.exceptions import (
    DocumentSearchException, 
    ServiceInitializationError,
    ResourceLimitError,
    log_exception
)

logger = logging.getLogger(__name__)

# Security scheme для майбутнього додавання автентифікації
security = HTTPBearer(auto_error=False)


class RequestContext:
    """
    Контекст запиту з корисною інформацією для логування та аналітики.
    
    Цей клас інкапсулює всю контекстну інформацію про поточний запит:
    - Ідентифікація клієнта
    - Timing метрики  
    - Request tracing
    - User context (для майбутньої автентифікації)
    
    Архітектурний патерн: Context Object
    Передається через всі шари додатку для збереження контексту.
    """
    
    def __init__(self, request: Request):
        self.request = request
        self.start_time = time.time()
        self.request_id = self._generate_request_id()
        self.client_ip = self._get_client_ip()
        self.user_agent = request.headers.get("user-agent", "unknown")
        self.correlation_id = request.headers.get("x-correlation-id")
        
        # User context (заповнюється при автентифікації)
        self.user_id: Optional[str] = None
        self.user_roles: Optional[list] = None
        
        # Request metrics
        self.metrics: Dict[str, Any] = {}
    
    def _generate_request_id(self) -> str:
        """Генерує унікальний ідентифікатор запиту."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_client_ip(self) -> str:
        """Витягує IP адресу клієнта з урахуванням proxy."""
        # Перевіряємо headers від reverse proxy
        forwarded_for = self.request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = self.request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback до прямого з'єднання
        if self.request.client:
            return self.request.client.host
        
        return "unknown"
    
    def add_metric(self, name: str, value: Any) -> None:
        """Додає метрику до контексту запиту."""
        self.metrics[name] = value
    
    def get_duration_ms(self) -> float:
        """Повертає тривалість запиту в мілісекундах."""
        return (time.time() - self.start_time) * 1000
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Конвертує контекст в словник для логування."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "user_id": self.user_id,
            "duration_ms": self.get_duration_ms(),
            "metrics": self.metrics
        }


# === Базові залежності ===

def get_request_context(request: Request) -> RequestContext:
    """
    Dependency для отримання контексту запиту.
    
    Ця залежність автоматично створює та ін'єктує RequestContext
    в будь-який endpoint, який її потребує.
    
    Usage:
    @app.get("/search")
    def search(context: RequestContext = Depends(get_request_context)):
        logger.info(f"Request from {context.client_ip}")
    """
    context = RequestContext(request)
    
    # Логуємо початок обробки запиту
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "extra_data": {
                "request_context": context.to_log_dict(),
                "request_method": request.method,
                "request_path": str(request.url.path),
                "query_params": dict(request.query_params)
            }
        }
    )
    
    return context


# === Сервісні залежності ===

def get_search_service():
    """
    Dependency для отримання search service.
    
    Перевіряє що сервіс правильно ініціалізований перед використанням.
    Якщо сервіс недоступний - кидає HTTP 503 (Service Unavailable).
    """
    try:
        # Перевіряємо базову працездатність сервісу
        stats = search_service.get_document_stats()
        if not stats.get("system_health", {}).get("vector_db_healthy", False):
            logger.error("Search service health check failed: vector DB unhealthy")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service is temporarily unavailable"
            )
        
        return search_service
        
    except DocumentSearchException as e:
        log_exception(logger, e, context={"dependency": "search_service"})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.get_user_message()
        )
    except Exception as e:
        logger.error(f"Unexpected error in search service dependency: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service initialization failed"
        )


def get_embedding_service():
    """
    Dependency для отримання embedding service.
    
    Перевіряє що ML модель завантажена та готова до роботи.
    """
    try:
        model_info = embedding_service.get_model_info()
        if not model_info.get("loaded", False):
            logger.error("Embedding service health check failed: model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI model is not ready. Please try again later."
            )
        
        return embedding_service
        
    except DocumentSearchException as e:
        log_exception(logger, e, context={"dependency": "embedding_service"})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.get_user_message()
        )
    except Exception as e:
        logger.error(f"Unexpected error in embedding service dependency: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable"
        )


def get_vector_store():
    """
    Dependency для отримання vector store service.
    
    Перевіряє підключення до векторної бази даних.
    """
    try:
        if not vector_store.health_check():
            logger.error("Vector store health check failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is temporarily unavailable"
            )
        
        return vector_store
        
    except DocumentSearchException as e:
        log_exception(logger, e, context={"dependency": "vector_store"})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.get_user_message()
        )
    except Exception as e:
        logger.error(f"Unexpected error in vector store dependency: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )


# === Rate Limiting та Security ===

class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    В production системі варто використовувати Redis-based rate limiting
    або зовнішній сервіс, але для початку цього достатньо.
    
    Архітектурний патерн: Token Bucket
    Кожен клієнт має "відро" з токенами, які поповнюються з фіксованою швидкістю.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, Dict[str, Any]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Перевіряє чи дозволений запит від клієнта."""
        now = time.time()
        
        if client_id not in self.clients:
            self.clients[client_id] = {
                "requests": 1,
                "window_start": now
            }
            return True
        
        client_data = self.clients[client_id]
        
        # Скидаємо лічильник якщо минуло вікно
        if now - client_data["window_start"] > self.window_seconds:
            client_data["requests"] = 1
            client_data["window_start"] = now
            return True
        
        # Перевіряємо ліміт
        if client_data["requests"] >= self.max_requests:
            return False
        
        # Інкрементуємо лічильник
        client_data["requests"] += 1
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Повертає кількість залишкових запитів."""
        if client_id not in self.clients:
            return self.max_requests
        
        return max(0, self.max_requests - self.clients[client_id]["requests"])


# Глобальний rate limiter
# В production це має бути Redis-based або вивантажено в окремий middleware
_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


def check_rate_limit(context: RequestContext = Depends(get_request_context)) -> None:
    """
    Dependency для перевірки rate limiting.
    
    Використовує IP адресу як ідентифікатор клієнта.
    В майбутньому можна замінити на user_id після автентифікації.
    """
    client_id = context.client_ip
    
    if not _rate_limiter.is_allowed(client_id):
        logger.warning(
            f"Rate limit exceeded for client {client_id}",
            extra={"extra_data": context.to_log_dict()}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
            headers={
                "Retry-After": str(_rate_limiter.window_seconds),
                "X-RateLimit-Limit": str(_rate_limiter.max_requests),
                "X-RateLimit-Remaining": str(_rate_limiter.get_remaining(client_id))
            }
        )


# === Валідаційні залежності ===

def validate_search_limits(
    limit: Optional[int] = None,
    score_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Dependency для валідації параметрів пошуку.
    
    Перевіряє та нормалізує параметри пошуку згідно з системними лімітами.
    Запобігає зловживанням та забезпечує стабільність системи.
    """
    validated_params = {}
    
    # Валідація limit
    if limit is not None:
        if limit <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be greater than 0"
            )
        if limit > 100:  # Максимальний ліміт для захисту системи
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit cannot exceed 100 results"
            )
        validated_params["limit"] = limit
    else:
        validated_params["limit"] = settings.default_limit
    
    # Валідація score_threshold
    if score_threshold is not None:
        if not 0.0 <= score_threshold <= 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Score threshold must be between 0.0 and 1.0"
            )
        validated_params["score_threshold"] = score_threshold
    else:
        validated_params["score_threshold"] = settings.similarity_threshold
    
    return validated_params


# === Автентифікація та авторизація (заготовка) ===

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    context: RequestContext = Depends(get_request_context)
) -> Optional[Dict[str, Any]]:
    """
    Dependency для автентифікації користувача.
    
    Поки що повертає None (відкритий доступ), але готова для
    додавання JWT токенів або іншої системи автентифікації.
    """
    if credentials is None:
        # Відкритий доступ - автентифікація не обов'язкова
        return None
    
    # TODO: Додати перевірку JWT токену
    # try:
    #     payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
    #     user_id = payload.get("sub")
    #     if user_id is None:
    #         raise HTTPException(status_code=401, detail="Invalid authentication")
    #     
    #     context.user_id = user_id
    #     return {"user_id": user_id, "roles": payload.get("roles", [])}
    # except JWTError:
    #     raise HTTPException(status_code=401, detail="Invalid authentication")
    
    # Поки що заглушка
    logger.debug("Authentication bypassed - open access mode")
    return None


def require_admin(current_user: Optional[Dict] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency для endpoints що потребують admin права.
    
    Поки що не використовується, але готова для майбутньої авторизації.
    """
    # TODO: Перевірити права користувача
    # if current_user is None:
    #     raise HTTPException(status_code=401, detail="Authentication required")
    # 
    # if "admin" not in current_user.get("roles", []):
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    # Поки що заглушка - всі мають admin права
    logger.debug("Authorization bypassed - open admin access")
    return current_user or {"user_id": "anonymous", "roles": ["admin"]}


# === Monitoring та Metrics ===

@asynccontextmanager
async def track_endpoint_metrics(
    endpoint_name: str,
    context: RequestContext
):
    """
    Async context manager для автоматичного трекінгу метрик endpoints.
    
    Використання:
    async with track_endpoint_metrics("search", context):
        result = await some_operation()
        # метрики автоматично записуються
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting endpoint: {endpoint_name}")
        yield
        
        # Успішне завершення
        duration_ms = (time.time() - start_time) * 1000
        context.add_metric(f"{endpoint_name}_duration_ms", duration_ms)
        context.add_metric(f"{endpoint_name}_status", "success")
        
        logger.info(
            f"Endpoint completed: {endpoint_name} in {duration_ms:.2f}ms",
            extra={"extra_data": context.to_log_dict()}
        )
        
    except Exception as e:
        # Обробка помилок
        duration_ms = (time.time() - start_time) * 1000
        context.add_metric(f"{endpoint_name}_duration_ms", duration_ms)
        context.add_metric(f"{endpoint_name}_status", "error")
        context.add_metric(f"{endpoint_name}_error_type", type(e).__name__)
        
        logger.error(
            f"Endpoint failed: {endpoint_name} after {duration_ms:.2f}ms - {str(e)}",
            extra={"extra_data": context.to_log_dict()}
        )
        
        raise  # Re-raise для обробки в endpoint


# === Комбіновані залежності для зручності ===

# Типові набори залежностей для різних типів endpoints
SearchDependencies = Annotated[
    tuple,
    Depends(lambda: (
        Depends(get_request_context),
        Depends(get_search_service),
        Depends(check_rate_limit),
        Depends(validate_search_limits)
    ))
]

AdminDependencies = Annotated[
    tuple,
    Depends(lambda: (
        Depends(get_request_context),
        Depends(require_admin),
        Depends(get_search_service)
    ))
]


# === Експорт основних залежностей ===

__all__ = [
    # Context
    "RequestContext",
    "get_request_context",
    
    # Services
    "get_search_service",
    "get_embedding_service", 
    "get_vector_store",
    
    # Security & Validation
    "check_rate_limit",
    "validate_search_limits",
    "get_current_user",
    "require_admin",
    
    # Monitoring
    "track_endpoint_metrics",
    
    # Combined
    "SearchDependencies",
    "AdminDependencies"
]