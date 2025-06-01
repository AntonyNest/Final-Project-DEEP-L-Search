# Logging configuration - система логування

"""
Система логування для Document Search Service.

Логування в ML системах критично важливе через:
1. **Debugging складних ML pipeline** - треба розуміти де саме виникла помилка
2. **Моніторинг продуктивності** - час обробки, використання ресурсів
3. **Аудит операцій** - хто що шукав, які документи індексувались
4. **Безпека** - виявлення підозрілої активності

Архітектурний підхід: Structured Logging
Використовуємо структуровані логи (JSON) для легкого парсингу системами моніторингу.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import settings


class JSONFormatter(logging.Formatter):
    """
    Custom formatter для виводу логів у JSON форматі.
    
    JSON логи мають кілька переваг:
    1. **Структурованість** - легко парсяти автоматично
    2. **Стандартизація** - єдиний формат для всіх компонентів
    3. **Розширюваність** - легко додавати нові поля
    4. **Інтеграція** - сумісність з ELK stack, Grafana, etc.
    
    Кожен лог запис містить:
    - timestamp: час події
    - level: рівень логування (INFO, ERROR, etc.)
    - logger: назва модуля що логує
    - message: основне повідомлення
    - extra_data: додаткові структуровані дані
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Конвертує LogRecord в JSON рядок.
        
        Цей метод викликається для кожного лог повідомлення
        та трансформує його в структурований JSON формат.
        """
        # Базова структура лог запису
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Додаємо інформацію про виключення, якщо є
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Додаємо додаткові дані, якщо передані через extra параметр
        if hasattr(record, 'extra_data'):
            log_entry["extra"] = record.extra_data
        
        # Додаємо контекстну інформацію для ML операцій
        if hasattr(record, 'operation_type'):
            log_entry["operation"] = {
                "type": record.operation_type,
                "duration_ms": getattr(record, 'duration_ms', None),
                "status": getattr(record, 'operation_status', None)
            }
        
        # Додаємо інформацію про користувача, якщо доступна
        if hasattr(record, 'user_context'):
            log_entry["user"] = record.user_context
        
        return json.dumps(log_entry, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """
    Filter для додавання контекстної інформації до логів.
    
    Цей фільтр автоматично додає корисну інформацію
    до кожного лог запису без необхідності вказувати її вручну.
    """
    
    def __init__(self, service_name: str = "document-search"):
        super().__init__()
        self.service_name = service_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Додає контекстну інформацію до лог запису.
        
        Цей метод викликається для кожного лог повідомлення
        перед його форматуванням та виводом.
        """
        # Додаємо назву сервісу
        record.service = self.service_name
        
        # Додаємо environment інформацію
        record.environment = "development"  # В production читати з конфігурації
        
        # Додаємо унікальний ID для трасування запитів (якщо доступний)
        # В реальній системі це може бути correlation ID з middleware
        if not hasattr(record, 'request_id'):
            record.request_id = None
        
        return True


class MLOperationLogger:
    """
    Спеціалізований logger для ML операцій.
    
    Цей клас надає зручний API для логування специфічних
    для ML системи подій з автоматичним збором метрик.
    
    Використання:
    ml_logger = MLOperationLogger("embedding_service")
    with ml_logger.operation("generate_embedding") as ctx:
        embedding = model.encode(text)
        ctx.add_metric("input_length", len(text))
        ctx.add_metric("embedding_dim", len(embedding))
    """
    
    def __init__(self, component_name: str):
        self.logger = logging.getLogger(f"ml.{component_name}")
        self.component_name = component_name
    
    def operation(self, operation_name: str):
        """
        Context manager для логування ML операцій з автоматичним вимірюванням часу.
        """
        return MLOperationContext(self.logger, operation_name, self.component_name)
    
    def log_model_info(self, model_name: str, model_params: Dict[str, Any]):
        """Логування інформації про завантажену ML модель."""
        self.logger.info(
            f"Model loaded: {model_name}",
            extra={
                "extra_data": {
                    "model_name": model_name,
                    "model_parameters": model_params,
                    "component": self.component_name
                }
            }
        )
    
    def log_performance_metrics(self, operation: str, metrics: Dict[str, float]):
        """Логування метрик продуктивності."""
        self.logger.info(
            f"Performance metrics for {operation}",
            extra={
                "extra_data": {
                    "operation": operation,
                    "metrics": metrics,
                    "component": self.component_name
                }
            }
        )


class MLOperationContext:
    """
    Context manager для окремої ML операції.
    
    Автоматично вимірює час виконання та дозволяє
    додавати метрики протягом виконання операції.
    """
    
    def __init__(self, logger: logging.Logger, operation_name: str, component_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.component_name = component_name
        self.start_time = None
        self.metrics = {}
        self.status = "started"
    
    def __enter__(self):
        """Початок операції."""
        import time
        self.start_time = time.time()
        
        self.logger.info(
            f"Starting {self.operation_name}",
            extra={
                "operation_type": self.operation_name,
                "operation_status": "started",
                "extra_data": {"component": self.component_name}
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершення операції з логуванням результатів."""
        import time
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is not None:
            self.status = "failed"
            self.logger.error(
                f"Failed {self.operation_name}: {exc_val}",
                extra={
                    "operation_type": self.operation_name,
                    "operation_status": "failed",
                    "duration_ms": duration_ms,
                    "extra_data": {
                        "component": self.component_name,
                        "metrics": self.metrics,
                        "error_type": exc_type.__name__
                    }
                },
                exc_info=True
            )
        else:
            self.status = "completed"
            self.logger.info(
                f"Completed {self.operation_name} in {duration_ms:.2f}ms",
                extra={
                    "operation_type": self.operation_name,
                    "operation_status": "completed",
                    "duration_ms": duration_ms,
                    "extra_data": {
                        "component": self.component_name,
                        "metrics": self.metrics
                    }
                }
            )
    
    def add_metric(self, name: str, value: Any):
        """Додає метрику до поточної операції."""
        self.metrics[name] = value
    
    def log_intermediate(self, message: str, **kwargs):
        """Логування проміжного стану операції."""
        self.logger.debug(
            f"{self.operation_name}: {message}",
            extra={
                "operation_type": self.operation_name,
                "operation_status": "in_progress",
                "extra_data": {
                    "component": self.component_name,
                    "intermediate_data": kwargs
                }
            }
        )


def setup_logging() -> None:
    """
    Головна функція налаштування системи логування.
    
    Ця функція має викликатись один раз при старті додатку
    для налаштування всіх логерів та handlers.
    
    Архітектурний підхід:
    1. **Hierarchical Loggers** - різні рівні деталізації для різних компонентів
    2. **Multiple Handlers** - одночасний вивід в консоль та файли
    3. **Structured Logging** - JSON формат для автоматичної обробки
    4. **Rotation** - автоматичне ротування лог файлів
    """
    
    # Отримуємо рівень логування з конфігурації
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Створюємо директорію для логів
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Очищуємо існуючі handlers (важливо для тестів)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Налаштовуємо root logger
    root_logger.setLevel(log_level)
    
    # === Console Handler для development ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Для консолі використовуємо простий формат у development
    if settings.log_level.upper() == "DEBUG":
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        # У production використовуємо JSON навіть для консолі
        console_formatter = JSONFormatter()
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)
    
    # === File Handler для production логів ===
    if settings.log_file:
        log_file_path = Path(settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler для автоматичного ротування
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,              # Зберігати 5 backup файлів
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())
        file_handler.addFilter(ContextFilter())
        root_logger.addHandler(file_handler)
    
    # === Спеціальні логери для різних компонентів ===
    
    # Logger для ML операцій (може мати інший рівень деталізації)
    ml_logger = logging.getLogger("ml")
    ml_logger.setLevel(logging.DEBUG)  # ML операції логуємо детально
    
    # Logger для API запитів
    api_logger = logging.getLogger("api")
    api_logger.setLevel(log_level)
    
    # Logger для векторної БД
    vector_db_logger = logging.getLogger("vector_db")
    vector_db_logger.setLevel(log_level)
    
    # Налаштування для зовнішніх бібліотек
    # Приглушуємо verbose логи від HTTP клієнтів
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Налаштування для ML бібліотек
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Повідомляємо про успішне налаштування
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging system initialized",
        extra={
            "extra_data": {
                "log_level": settings.log_level,
                "log_file": str(settings.log_file) if settings.log_file else None,
                "handlers_count": len(root_logger.handlers)
            }
        }
    )


def get_ml_logger(component_name: str) -> MLOperationLogger:
    """
    Factory function для створення ML логерів.
    
    Використовуйте цю функцію для отримання спеціалізованого
    логера для ML компонентів.
    
    Приклад використання:
    logger = get_ml_logger("embedding_service")
    with logger.operation("encode_batch") as ctx:
        embeddings = model.encode(texts)
        ctx.add_metric("batch_size", len(texts))
    """
    return MLOperationLogger(component_name)


# === Utility функції для логування в різних контекстах ===

def log_api_request(logger: logging.Logger, method: str, path: str, user_id: Optional[str] = None):
    """Утиліта для логування API запитів."""
    extra_data = {
        "api_request": {
            "method": method,
            "path": path,
            "user_id": user_id
        }
    }
    
    logger.info(
        f"API Request: {method} {path}",
        extra={"extra_data": extra_data}
    )


def log_search_query(logger: logging.Logger, query: str, results_count: int, duration_ms: float):
    """Утиліта для логування пошукових запитів."""
    extra_data = {
        "search_query": {
            "query_length": len(query),
            "query_hash": hash(query),  # Хеш для приватності
            "results_count": results_count,
            "duration_ms": duration_ms
        }
    }
    
    logger.info(
        f"Search completed: {results_count} results in {duration_ms:.2f}ms",
        extra={"extra_data": extra_data}
    )


def log_document_processing(logger: logging.Logger, file_path: str, chunks_count: int, status: str):
    """Утиліта для логування обробки документів."""
    extra_data = {
        "document_processing": {
            "file_name": Path(file_path).name,
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024) if Path(file_path).exists() else 0,
            "chunks_count": chunks_count,
            "status": status
        }
    }
    
    logger.info(
        f"Document processed: {Path(file_path).name} -> {chunks_count} chunks ({status})",
        extra={"extra_data": extra_data}
    )


# Експортуємо основні функції для зручності використання
__all__ = [
    "setup_logging",
    "get_ml_logger", 
    "MLOperationLogger",
    "log_api_request",
    "log_search_query",
    "log_document_processing"
]