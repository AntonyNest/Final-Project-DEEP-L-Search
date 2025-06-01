# Search API endpoints

"""
Search API Endpoints - Інтерфейси для семантичного пошуку документів.

Ці endpoints забезпечують:
1. **Семантичний пошук** - основна функція пошуку по змісту
2. **Автокомпліт/Саджешн** - допомога користувачам в формуванні запитів
3. **Схожі документи** - пошук релевантних документів до заданого
4. **Аналітику пошуків** - статистика та insights

Архітектурний підхід: RESTful API Design
- GET для отримання даних (пошук, статистика)
- POST для складних запитів з body (детальний пошук)
- Консистентна структура відповідей
- Proper HTTP status codes

Кожен endpoint слідує pattern:
1. Валідація вхідних даних (через Pydantic)
2. Бізнес-логіка (делегування до сервісів)
3. Форматування відповіді (через схеми)
4. Логування та метрики
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    SearchRequest, SearchResponse, SearchResultItem, 
    BaseResponse, SystemStatsResponse, ErrorResponse, ErrorDetail
)
from app.api.dependencies import (
    RequestContext, get_request_context,
    get_search_service, get_embedding_service,
    check_rate_limit, validate_search_limits,
    track_endpoint_metrics
)
from app.utils.exceptions import (
    DocumentSearchException, SearchQueryError, 
    ValidationError, log_exception
)

# Створюємо router для всіх search endpoints
router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/semantic",
    response_model=SearchResponse,
    summary="Семантичний пошук документів",
    description="""
    Виконує інтелектуальний семантичний пошук по колекції документів 
    з використанням AI ембедингів.
    
    **Особливості:**
    - Розуміє синоніми та контекст (не тільки точні співпадіння)
    - Підтримує українську та російську мови
    - Ранжування за релевантністю
    - Фільтрація за метаданими документів
    - Детальна статистика виконання
    
    **Приклади запитів:**
    - "інформаційне забезпечення системи управління"
    - "архітектура платформи штучного інтелекту"
    - "технічні вимоги безпеки даних"
    """,
    responses={
        200: {"description": "Успішний пошук"},
        400: {"description": "Некоректний запит"},
        429: {"description": "Перевищено ліміт запитів"},
        503: {"description": "Сервіс тимчасово недоступний"}
    }
)
async def semantic_search(
    search_request: SearchRequest,
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _rate_limit = Depends(check_rate_limit)
):
    """
    Основний endpoint для семантичного пошуку.
    
    Цей endpoint реалізує повний ML pipeline:
    1. Валідація запиту
    2. Генерація ембединга для пошукового запиту
    3. Векторний пошук у базі даних
    4. Пост-обробка та ранжування результатів
    5. Форматування відповіді
    """
    async with track_endpoint_metrics("semantic_search", context):
        try:
            # Логуємо пошуковий запит (без чутливих даних)
            logger.info(
                f"Semantic search request",
                extra={
                    "extra_data": {
                        "query_length": len(search_request.query),
                        "query_hash": hash(search_request.query),  # Хеш для privacy
                        "limit": search_request.limit,
                        "score_threshold": search_request.score_threshold,
                        "filters_count": len(search_request.file_types or []),
                        "request_context": context.to_log_dict()
                    }
                }
            )
            
            # Валідація запиту
            if not search_request.query.strip():
                raise SearchQueryError(
                    query=search_request.query,
                    reason="Query cannot be empty",
                    suggestions=["Введіть ключові слова для пошуку", "Спробуйте більш конкретний запит"]
                )
            
            if len(search_request.query) > 1000:
                raise SearchQueryError(
                    query=search_request.query,
                    reason="Query too long",
                    suggestions=["Скоротіть запит до 1000 символів"]
                )
            
            # Підготовка фільтрів для пошуку
            filters = {}
            
            if search_request.file_types:
                # Конвертуємо enum значення в рядки
                file_extensions = [ft.value for ft in search_request.file_types]
                filters["file_extension"] = {"range": {"gte": file_extensions[0]}}  # Simplified for demo
            
            if search_request.source_files:
                filters["file_name"] = search_request.source_files[0]  # Simplified - тільки перший файл
            
            # Виконуємо пошук через search service
            search_result = search_service.search(
                query=search_request.query,
                limit=search_request.limit,
                score_threshold=search_request.score_threshold,
                filters=filters if filters else None,
                include_stats=search_request.include_stats
            )
            
            # Перевіряємо результат пошуку
            if not search_result["success"]:
                raise DocumentSearchException(
                    message=search_result.get("message", "Search failed"),
                    error_code="SEARCH_FAILED"
                )
            
            # Додаємо метрики до контексту
            if search_result.get("stats"):
                context.add_metric("search_results_count", len(search_result["results"]))
                context.add_metric("search_duration_ms", search_result["stats"]["total_time_ms"])
                context.add_metric("embedding_duration_ms", search_result["stats"]["embedding_time_ms"])
            
            # Конвертуємо результати в API формат
            formatted_results = []
            for result_data in search_result["results"]:
                # Створюємо SearchResultItem з raw даних
                formatted_result = SearchResultItem(
                    chunk_id=result_data["chunk_id"],
                    text=result_data["text"],
                    score=result_data["score"],
                    source_file=result_data["source_file"],
                    chunk_index=result_data.get("chunk_index"),
                    metadata=result_data["metadata"],
                    highlighted_text=result_data.get("highlighted_text"),
                    keyword_matches=result_data.get("keyword_matches")
                )
                formatted_results.append(formatted_result)
            
            # Формуємо відповідь
            response = SearchResponse(
                success=True,
                message=f"Found {len(formatted_results)} results",
                query=search_request.query,
                results=formatted_results,
                stats=search_result.get("stats") if search_request.include_stats else None
            )
            
            return response
            
        except SearchQueryError as e:
            # Специфічні помилки запиту
            logger.warning(f"Search query error: {e.message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.get_user_message()
            )
            
        except DocumentSearchException as e:
            # Наші кастомні помилки
            log_exception(logger, e, context={"endpoint": "semantic_search"})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e.get_user_message()
            )
            
        except Exception as e:
            # Непередбачені помилки
            logger.error(f"Unexpected error in semantic search: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during search"
            )


@router.get(
    "/quick",
    response_model=SearchResponse,
    summary="Швидкий пошук (GET endpoint)",
    description="""
    Спрощений endpoint для швидкого пошуку через GET запит.
    Підходить для простих пошукових запитів без складних фільтрів.
    
    **Використання:**
    - Інтеграція з веб-формами
    - Тестування через браузер
    - Simple API клієнти
    """
)
async def quick_search(
    q: str = Query(..., description="Пошуковий запит", min_length=1, max_length=500),
    limit: int = Query(10, description="Кількість результатів", ge=1, le=50),
    threshold: float = Query(0.5, description="Поріг схожості", ge=0.0, le=1.0),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _rate_limit = Depends(check_rate_limit)
):
    """
    GET endpoint для простого пошуку без складної структури запиту.
    
    Корисний для швидкого тестування та простих інтеграцій.
    """
    async with track_endpoint_metrics("quick_search", context):
        try:
            # Конвертуємо GET параметри в SearchRequest
            search_request = SearchRequest(
                query=q,
                limit=limit,
                score_threshold=threshold,
                include_stats=True  # Завжди включаємо stats для GET
            )
            
            # Викликаємо основну логіку semantic_search
            # Створюємо mock dependencies для internal call
            search_result = search_service.search(
                query=search_request.query,
                limit=search_request.limit,
                score_threshold=search_request.score_threshold,
                include_stats=True
            )
            
            if not search_result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=search_result.get("message", "Search failed")
                )
            
            # Формуємо спрощену відповідь
            formatted_results = []
            for result_data in search_result["results"]:
                formatted_result = SearchResultItem(
                    chunk_id=result_data["chunk_id"],
                    text=result_data["text"],
                    score=result_data["score"],
                    source_file=result_data["source_file"],
                    metadata=result_data["metadata"]
                )
                formatted_results.append(formatted_result)
            
            return SearchResponse(
                success=True,
                message=f"Quick search: {len(formatted_results)} results",
                query=q,
                results=formatted_results,
                stats=search_result.get("stats")
            )
            
        except HTTPException:
            raise  # Перекидаємо HTTP exceptions без змін
        except Exception as e:
            logger.error(f"Quick search error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Quick search failed"
            )


@router.get(
    "/suggestions",
    response_model=Dict[str, List[str]],
    summary="Автокомпліт та саджешни для пошуку",
    description="""
    Повертає список саджешнів для автокомпліту пошукових запитів.
    
    **Функціональність:**
    - Популярні пошукові терміни
    - Саджешни на основі контенту документів
    - Виправлення опечаток (майбутня функція)
    """
)
async def get_search_suggestions(
    partial_query: str = Query("", description="Частковий запит для саджешнів", max_length=100),
    limit: int = Query(10, description="Кількість саджешнів", ge=1, le=20),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service)
):
    """
    Endpoint для отримання саджешнів пошукових запитів.
    
    На поточному етапі повертає статичні саджешни.
    В майбутньому можна додати ML-based саджешни на основі:
    - Аналізу контенту документів
    - Історії пошуків
    - NLP обробки запитів
    """
    async with track_endpoint_metrics("search_suggestions", context):
        # Статичні саджешни для демонстрації
        # В production це мало б бути динамічно згенероване
        base_suggestions = [
            "інформаційне забезпечення",
            "архітектура системи",
            "штучний інтелект",
            "безпека даних",
            "технічні вимоги",
            "управління документами",
            "платформа ШІ",
            "векторна база даних",
            "машинне навчання",
            "обробка документів"
        ]
        
        # Фільтруємо саджешни на основі часткового запиту
        if partial_query:
            filtered_suggestions = [
                s for s in base_suggestions 
                if partial_query.lower() in s.lower()
            ]
        else:
            filtered_suggestions = base_suggestions
        
        # Обмежуємо кількість результатів
        limited_suggestions = filtered_suggestions[:limit]
        
        # Додаємо контекстні саджешни
        contextual_suggestions = []
        if "архітект" in partial_query.lower():
            contextual_suggestions = ["архітектура додатків", "архітектурне рішення", "архітектурні патерни"]
        elif "безпек" in partial_query.lower():
            contextual_suggestions = ["безпека системи", "кібербезпека", "інформаційна безпека"]
        
        return {
            "suggestions": limited_suggestions,
            "contextual": contextual_suggestions[:5],
            "partial_query": partial_query,
            "total_found": len(filtered_suggestions)
        }


@router.get(
    "/similar/{document_id}",
    response_model=SearchResponse,
    summary="Пошук схожих документів",
    description="""
    Знаходить документи схожі на заданий документ або фрагмент.
    
    **Використання:**
    - "Знайти схожі документи"
    - Рекомендації релевантного контенту
    - Дублікати та близькі за змістом документи
    """
)
async def find_similar_documents(
    document_id: str,
    limit: int = Query(10, description="Кількість схожих документів", ge=1, le=50),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service)
):
    """
    Пошук документів схожих на заданий.
    
    Використовує ембединг заданого документа для пошуку схожих.
    """
    async with track_endpoint_metrics("similar_documents", context):
        try:
            # TODO: Реалізувати логіку пошуку схожих документів
            # 1. Отримати ембединг документа за document_id
            # 2. Використати цей ембединг для векторного пошуку
            # 3. Виключити оригінальний документ з результатів
            
            # Поки що заглушка
            logger.info(f"Similar documents request for: {document_id}")
            
            return SearchResponse(
                success=True,
                message="Similar documents feature coming soon",
                results=[],
                stats=None
            )
            
        except Exception as e:
            logger.error(f"Similar documents search error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Similar documents search failed"
            )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Статистика пошукової системи",
    description="""
    Повертає детальну статистику про стан пошукової системи.
    
    **Включає:**
    - Кількість проіндексованих документів
    - Статистику векторної бази даних
    - Інформацію про ML модель
    - Здоров'я системи
    """
)
async def get_search_stats(
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service)
):
    """
    Endpoint для отримання статистики системи.
    
    Корисний для:
    - Моніторингу стану системи
    - Admin dashboards
    - Debugging та діагностики
    """
    async with track_endpoint_metrics("search_stats", context):
        try:
            stats = search_service.get_document_stats()
            
            return SystemStatsResponse(
                success=True,
                message="System statistics retrieved successfully",
                **stats  # Розпаковуємо всі статистики
            )
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system statistics"
            )


@router.post(
    "/analyze",
    response_model=Dict[str, Any],
    summary="Аналіз пошукового запиту",
    description="""
    Аналізує пошуковий запит без виконання пошуку.
    
    **Повертає:**
    - Ключові слова та терміни
    - Мовний аналіз
    - Рекомендації для покращення запиту
    - Прогнозовану кількість результатів
    """
)
async def analyze_query(
    query: str,
    context: RequestContext = Depends(get_request_context),
    embedding_service = Depends(get_embedding_service)
):
    """
    Аналіз запиту для допомоги користувачу в формуванні кращих пошукових запитів.
    
    Цей endpoint може бути корисний для:
    - Показу user-friendly insights про запит
    - Попереднього аналізу перед пошуком
    - Рекомендацій для покращення результатів
    """
    async with track_endpoint_metrics("analyze_query", context):
        try:
            # Базовий аналіз запиту
            analysis = {
                "query": query,
                "query_length": len(query),
                "word_count": len(query.split()),
                "estimated_complexity": "simple" if len(query.split()) <= 3 else "complex",
                "language": "uk" if any(char in query for char in "іїєґ") else "unknown",
                "keywords": query.split(),  # Спрощений розбір на слова
                "recommendations": []
            }
            
            # Додаємо рекомендації
            if len(query) < 10:
                analysis["recommendations"].append("Спробуйте більш детальний запит для кращих результатів")
            
            if len(query.split()) > 10:
                analysis["recommendations"].append("Занадто довгий запит може зменшити точність пошуку")
            
            # Генеруємо ембединг для технічного аналізу
            try:
                embedding = embedding_service.encode_single(query)
                analysis["embedding_generated"] = True
                analysis["embedding_dimension"] = len(embedding)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for analysis: {str(e)}")
                analysis["embedding_generated"] = False
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Query analysis failed"
            )


# === Health Check специфічний для search endpoints ===

@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Health check пошукової системи",
    description="Перевірка здоров'я всіх компонентів пошукової системи"
)
async def search_health_check(
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    embedding_service = Depends(get_embedding_service)
):
    """
    Детальна перевірка здоров'я пошукової системи.
    
    Перевіряє:
    - Доступність ML моделі
    - Підключення до векторної БД
    - Швидкодію основних операцій
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Перевірка embedding service
        try:
            model_info = embedding_service.get_model_info()
            health_status["components"]["embedding_service"] = {
                "status": "healthy" if model_info.get("loaded") else "unhealthy",
                "model_name": model_info.get("model_name"),
                "loaded": model_info.get("loaded", False)
            }
        except Exception as e:
            health_status["components"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Перевірка search service
        try:
            stats = search_service.get_document_stats()
            health_status["components"]["search_service"] = {
                "status": "healthy",
                "indexed_documents": stats.get("vector_database", {}).get("points_count", 0)
            }
        except Exception as e:
            health_status["components"]["search_service"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        # Загальний статус
        all_healthy = all(
            comp.get("status") == "healthy" 
            for comp in health_status["components"].values()
        )
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }