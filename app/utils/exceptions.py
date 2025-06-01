# Custom exceptions

"""
Custom Exceptions для Document Search Service.

Система кастомних виключень критично важлива для ML додатків через:
1. **Специфічні помилки ML pipeline** - embedding генерація, векторні операції
2. **Детальна діагностика** - точне розуміння де і чому виникла помилка  
3. **Graceful degradation** - різна обробка різних типів помилок
4. **User Experience** - зрозумілі повідомлення користувачу

Архітектурний принцип: Exception Hierarchy
Створюємо ієрархію виключень від загальних до специфічних, що дозволяє
catch-ити помилки на правильному рівні абстракції.

Структура ієрархії:
DocumentSearchException (базовий)
├── ConfigurationError (проблеми конфігурації)
├── MLServiceError (помилки ML компонентів)
│   ├── EmbeddingServiceError
│   ├── VectorStoreError  
│   └── ModelLoadError
├── DocumentProcessingError (помилки обробки документів)
│   ├── DocumentNotFoundError
│   ├── UnsupportedFormatError
│   └── ExtractionError
└── ValidationError (помилки валідації даних)
"""

from typing import Optional, Dict, Any, List
import traceback


class DocumentSearchException(Exception):
    """
    Базовий клас для всіх кастомних виключень в системі.
    
    Цей клас встановлює єдиний контракт для всіх наших виключень:
    - Зберігає детальну інформацію про помилку
    - Надає методи для логування та серіалізації
    - Підтримує context information для debugging
    
    Кожне наше виключення успадковується від цього класу,
    що дозволяє catch-ити всі системні помилки одним except блоком.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Ініціалізація базового виключення.
        
        Args:
            message: Основне повідомлення про помилку (для користувача)
            error_code: Унікальний код помилки (для програмної обробки)
            details: Додаткові деталі помилки (для debugging)
            cause: Оригінальне виключення що спричинило цю помилку
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.cause = cause
        
        # Автоматично додаємо traceback для debugging
        self.traceback_str = traceback.format_exc()
        
        # Додаємо інформацію про оригінальну причину
        if cause:
            self.details["caused_by"] = {
                "type": type(cause).__name__,
                "message": str(cause)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертує виключення в словник для API відповіді.
        
        Цей метод дозволяє легко серіалізувати помилку в JSON
        для відправки клієнту через API.
        """
        return {
            "error_code": self.error_code,
            "error_message": self.message,
            "error_type": self.__class__.__name__,
            "details": self.details
        }
    
    def get_user_message(self) -> str:
        """
        Повертає повідомлення, безпечне для показу користувачу.
        
        Видаляє технічні деталі, які можуть бути конфіденційними
        або незрозумілими для кінцевого користувача.
        """
        return self.message
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Повертає повну діагностичну інформацію для розробників.
        
        Ця інформація включає stack trace, деталі помилки,
        та всю контекстну інформацію для debugging.
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "traceback": self.traceback_str,
            "cause": str(self.cause) if self.cause else None
        }


# === Помилки конфігурації та ініціалізації ===

class ConfigurationError(DocumentSearchException):
    """
    Помилки в конфігурації системи.
    
    Ці помилки зазвичай виникають при старті додатку
    через неправильні налаштування або відсутні ресурси.
    """
    
    def __init__(self, message: str, config_field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        if config_field:
            self.details["config_field"] = config_field


class ServiceInitializationError(DocumentSearchException):
    """
    Помилки ініціалізації сервісів системи.
    
    Виникають коли не вдається запустити критично важливі
    компоненти системи (БД, ML моделі, etc.).
    """
    
    def __init__(self, service_name: str, message: str, **kwargs):
        super().__init__(
            f"Failed to initialize {service_name}: {message}",
            error_code="SERVICE_INIT_ERROR",
            **kwargs
        )
        self.details["service_name"] = service_name


# === Помилки ML компонентів ===

class MLServiceError(DocumentSearchException):
    """
    Базовий клас для помилок ML компонентів.
    
    Всі помилки пов'язані з машинним навчанням наслідуються
    від цього класу для легшої категоризації.
    """
    pass


class ModelLoadError(MLServiceError):
    """
    Помилки завантаження ML моделей.
    
    Ці помилки виникають при:
    - Відсутності файлів моделі
    - Несумісності версій
    - Недостатній пам'яті для завантаження
    """
    
    def __init__(self, model_name: str, reason: str, **kwargs):
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            **kwargs
        )
        self.details.update({
            "model_name": model_name,
            "failure_reason": reason
        })
    
    def get_user_message(self) -> str:
        """Спрощене повідомлення для користувача."""
        return "Сервіс тимчасово недоступний через проблеми з ML моделлю. Спробуйте пізніше."


class EmbeddingServiceError(MLServiceError):
    """
    Помилки сервісу генерації ембедингів.
    
    Виникають при:
    - Помилках векторизації тексту
    - Timeout операцій з моделлю
    - Некоректних вхідних даних
    """
    
    def __init__(self, operation: str, reason: str, input_data_info: Optional[Dict] = None, **kwargs):
        super().__init__(
            f"Embedding service failed during {operation}: {reason}",
            error_code="EMBEDDING_ERROR",
            **kwargs
        )
        self.details.update({
            "operation": operation,
            "failure_reason": reason
        })
        if input_data_info:
            self.details["input_data"] = input_data_info


class VectorStoreError(MLServiceError):
    """
    Помилки роботи з векторною базою даних.
    
    Включають:
    - Помилки підключення до Qdrant
    - Операції індексації/пошуку
    - Проблеми з колекціями
    """
    
    def __init__(self, operation: str, reason: str, collection_name: Optional[str] = None, **kwargs):
        super().__init__(
            f"Vector store operation '{operation}' failed: {reason}",
            error_code="VECTOR_STORE_ERROR", 
            **kwargs
        )
        self.details.update({
            "operation": operation,
            "failure_reason": reason
        })
        if collection_name:
            self.details["collection_name"] = collection_name
    
    def get_user_message(self) -> str:
        """Спрощене повідомлення для користувача."""
        return "Проблема з базою даних пошуку. Спробуйте пізніше або зверніться до адміністратора."


# === Помилки обробки документів ===

class DocumentProcessingError(DocumentSearchException):
    """
    Базовий клас для помилок обробки документів.
    
    Всі помилки пов'язані з читанням, парсингом та обробкою
    документів наслідуються від цього класу.
    """
    pass


class DocumentNotFoundError(DocumentProcessingError):
    """
    Помилка коли документ не знайдено.
    
    Може виникати при:
    - Видаленні файлу під час обробки
    - Неправильному шляху до файлу
    - Проблемах з правами доступу
    """
    
    def __init__(self, file_path: str, **kwargs):
        super().__init__(
            f"Document not found: {file_path}",
            error_code="DOCUMENT_NOT_FOUND",
            **kwargs
        )
        self.details["file_path"] = file_path
    
    def get_user_message(self) -> str:
        return "Запитаний документ не знайдено. Він міг бути переміщений або видалений."


class UnsupportedFormatError(DocumentProcessingError):
    """
    Помилка непідтримуваного формату документа.
    
    Виникає при спробі обробити файл в форматі,
    який система не вміє читати.
    """
    
    def __init__(self, file_path: str, file_format: str, supported_formats: List[str], **kwargs):
        super().__init__(
            f"Unsupported document format '{file_format}' for file {file_path}. "
            f"Supported formats: {', '.join(supported_formats)}",
            error_code="UNSUPPORTED_FORMAT",
            **kwargs
        )
        self.details.update({
            "file_path": file_path,
            "detected_format": file_format,
            "supported_formats": supported_formats
        })
    
    def get_user_message(self) -> str:
        supported = ", ".join(self.details["supported_formats"])
        return f"Формат файлу не підтримується. Підтримувані формати: {supported}"


class ExtractionError(DocumentProcessingError):
    """
    Помилки екстракції тексту з документів.
    
    Виникають при:
    - Пошкоджених файлах
    - Захищених паролем документах
    - Помилках бібліотек парсингу
    """
    
    def __init__(self, file_path: str, extraction_method: str, reason: str, **kwargs):
        super().__init__(
            f"Failed to extract text from {file_path} using {extraction_method}: {reason}",
            error_code="EXTRACTION_ERROR",
            **kwargs
        )
        self.details.update({
            "file_path": file_path,
            "extraction_method": extraction_method,
            "failure_reason": reason
        })
    
    def get_user_message(self) -> str:
        return "Не вдалося прочитати вміст документа. Файл може бути пошкоджений або захищений."


# === Помилки валідації та вхідних даних ===

class ValidationError(DocumentSearchException):
    """
    Помилки валідації вхідних даних.
    
    Виникають при некоректних параметрах запиту,
    неправильному форматі даних, тощо.
    """
    
    def __init__(self, field_name: str, field_value: Any, reason: str, **kwargs):
        super().__init__(
            f"Validation failed for field '{field_name}': {reason}",
            error_code="VALIDATION_ERROR",
            **kwargs
        )
        self.details.update({
            "field_name": field_name,
            "field_value": str(field_value),
            "validation_reason": reason
        })


class SearchQueryError(ValidationError):
    """
    Помилки в пошукових запитах.
    
    Спеціалізована валідаційна помилка для пошукових запитів
    з додатковою контекстною інформацією.
    """
    
    def __init__(self, query: str, reason: str, suggestions: Optional[List[str]] = None, **kwargs):
        super().__init__(
            field_name="query",
            field_value=query,
            reason=reason,
            **kwargs
        )
        self.error_code = "SEARCH_QUERY_ERROR"
        if suggestions:
            self.details["suggestions"] = suggestions
    
    def get_user_message(self) -> str:
        message = f"Помилка в пошуковому запиті: {self.details['validation_reason']}"
        if "suggestions" in self.details:
            message += f" Спробуйте: {', '.join(self.details['suggestions'])}"
        return message


# === Помилки ресурсів та обмежень ===

class ResourceLimitError(DocumentSearchException):
    """
    Помилки перевищення лімітів ресурсів.
    
    Виникають при:
    - Перевищенні лімітів API
    - Недостатній пам'яті
    - Timeout операцій
    """
    
    def __init__(self, resource_type: str, limit_value: Any, current_value: Any, **kwargs):
        super().__init__(
            f"Resource limit exceeded for {resource_type}: {current_value} > {limit_value}",
            error_code="RESOURCE_LIMIT_ERROR",
            **kwargs
        )
        self.details.update({
            "resource_type": resource_type,
            "limit_value": limit_value,
            "current_value": current_value
        })
    
    def get_user_message(self) -> str:
        return f"Перевищено ліміт для {self.details['resource_type']}. Спробуйте зменшити розмір запиту."


class TimeoutError(DocumentSearchException):
    """
    Помилки таймауту операцій.
    
    Критично важливо для ML операцій, які можуть виконуватись довго.
    """
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            error_code="TIMEOUT_ERROR",
            **kwargs
        )
        self.details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds
        })
    
    def get_user_message(self) -> str:
        return "Операція займає занадто багато часу. Спробуйте пізніше або зменшіть розмір запиту."


# === Утилітарні функції для роботи з виключеннями ===

def handle_external_exception(
    external_exc: Exception,
    operation_context: str,
    fallback_message: str = "An unexpected error occurred"
) -> DocumentSearchException:
    """
    Конвертує зовнішні виключення в наші кастомні.
    
    Ця функція дозволяє обгорнути помилки з зовнішніх бібліотек
    (PyPDF2, transformers, qdrant-client) в наш формат.
    
    Args:
        external_exc: Оригінальне виключення з зовнішньої бібліотеки
        operation_context: Контекст де виникла помилка
        fallback_message: Повідомлення якщо не вдається класифікувати помилку
        
    Returns:
        Відповідне кастомне виключення з збереженою інформацією
    """
    exc_type = type(external_exc).__name__
    exc_message = str(external_exc)
    
    # Намагаємося класифікувати помилку за типом та повідомленням
    if "connection" in exc_message.lower() or "timeout" in exc_message.lower():
        if "qdrant" in operation_context.lower():
            return VectorStoreError(
                operation=operation_context,
                reason=f"Connection issue: {exc_message}",
                cause=external_exc
            )
    
    elif "memory" in exc_message.lower() or "cuda" in exc_message.lower():
        return MLServiceError(
            f"Resource issue in {operation_context}: {exc_message}",
            error_code="RESOURCE_ERROR",
            cause=external_exc
        )
    
    elif "file" in exc_message.lower() and "not found" in exc_message.lower():
        return DocumentNotFoundError(
            file_path="unknown",
            cause=external_exc
        )
    
    # Якщо не вдається класифікувати - повертаємо загальну помилку
    return DocumentSearchException(
        message=f"{fallback_message}: {exc_message}",
        error_code="EXTERNAL_ERROR",
        details={
            "external_exception_type": exc_type,
            "operation_context": operation_context
        },
        cause=external_exc
    )


def log_exception(logger, exception: DocumentSearchException, context: Optional[Dict[str, Any]] = None):
    """
    Логування виключення з усією контекстною інформацією.
    
    Ця функція забезпечує консистентне логування всіх помилок
    з максимальною діагностичною інформацією.
    """
    log_data = exception.get_debug_info()
    if context:
        log_data["context"] = context
    
    logger.error(
        f"Exception occurred: {exception.message}",
        extra={"extra_data": log_data}
    )


# === Декоратори для автоматичної обробки помилок ===

def handle_service_errors(operation_name: str):
    """
    Декоратор для автоматичної обробки помилок в сервісах.
    
    Використання:
    @handle_service_errors("document_processing")
    def process_document(self, file_path):
        # код може кинути будь-яке виключення
        # декоратор автоматично конвертує його в наш формат
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DocumentSearchException:
                # Наші кастомні виключення пропускаємо без змін
                raise
            except Exception as e:
                # Зовнішні виключення конвертуємо
                custom_exc = handle_external_exception(
                    e, 
                    operation_context=operation_name,
                    fallback_message=f"Error in {operation_name}"
                )
                raise custom_exc from e
        return wrapper
    return decorator


# Експортуємо основні класи виключень
__all__ = [
    # Базові класи
    "DocumentSearchException",
    "MLServiceError", 
    "DocumentProcessingError",
    "ValidationError",
    
    # Конфігурація та ініціалізація
    "ConfigurationError",
    "ServiceInitializationError",
    
    # ML помилки
    "ModelLoadError",
    "EmbeddingServiceError", 
    "VectorStoreError",
    
    # Обробка документів
    "DocumentNotFoundError",
    "UnsupportedFormatError",
    "ExtractionError",
    
    # Валідація
    "SearchQueryError",
    
    # Ресурси
    "ResourceLimitError",
    "TimeoutError",
    
    # Утилітарні функції
    "handle_external_exception",
    "log_exception",
    "handle_service_errors"
]