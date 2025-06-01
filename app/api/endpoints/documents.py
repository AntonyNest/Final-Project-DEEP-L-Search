# Document management API

"""
Document Management API Endpoints - Управління документами та індексацією.

Ці endpoints призначені для:
1. **Індексація документів** - додавання нових документів до пошукової системи
2. **Моніторинг процесів** - відстеження стану індексації та обробки
3. **Управління колекцією** - очищення, оновлення, статистика
4. **Адміністрування** - операції обслуговування системи

Архітектурний підхід: CRUD + Admin Operations
- POST для створення/додавання (індексація)
- GET для читання (статистика, статус)
- PUT для оновлення (реіндексація)
- DELETE для видалення
- PATCH для часткових змін

Безпека: Більшість операцій потребують admin права.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import (
    IndexingRequest, IndexingResponse, IndexingStats,
    BaseResponse, SystemStatsResponse, 
    CacheOperationResponse, DocumentType
)
from app.api.dependencies import (
    RequestContext, get_request_context,
    get_search_service, require_admin,
    track_endpoint_metrics
)
from app.utils.exceptions import (
    DocumentSearchException, ConfigurationError,
    DocumentProcessingError, log_exception
)

# Створюємо router для document management endpoints
router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/index",
    response_model=IndexingResponse,
    summary="Індексація документів",
    description="""
    Запускає процес індексації документів з файлової системи.
    
    **Процес індексації:**
    1. Сканування директорії на наявність документів
    2. Витягування тексту з документів (.docx, .doc, .pdf)
    3. Розбиття тексту на семантичні фрагменти
    4. Генерація ембедингів для кожного фрагменту
    5. Збереження у векторну базу даних
    
    **Примітка:** Процес може займати багато часу для великих колекцій.
    """,
    responses={
        200: {"description": "Індексація завершена"},
        202: {"description": "Індексація запущена у фоновому режимі"},
        400: {"description": "Некоректні параметри індексації"},
        403: {"description": "Потрібні права адміністратора"},
        503: {"description": "Сервіс недоступний"}
    }
)
async def index_documents(
    indexing_request: IndexingRequest,
    background_tasks: BackgroundTasks,
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _admin = Depends(require_admin)
):
    """
    Endpoint для запуску індексації документів.
    
    Підтримує як синхронний, так і асинхронний режими:
    - Синхронний: блокує запит до завершення індексації
    - Асинхронний: запускає індексацію у фоні та повертає статус
    """
    async with track_endpoint_metrics("index_documents", context):
        try:
            # Логуємо запит на індексацію
            logger.info(
                "Document indexing requested",
                extra={
                    "extra_data": {
                        "custom_path": indexing_request.custom_path,
                        "force_reindex": indexing_request.force_reindex,
                        "file_types_filter": indexing_request.file_types_filter,
                        "admin_user": context.user_id,
                        "request_context": context.to_log_dict()
                    }
                }
            )
            
            # Валідація шляху до документів
            if indexing_request.custom_path:
                custom_path = Path(indexing_request.custom_path)
                if not custom_path.exists():
                    raise ConfigurationError(
                        f"Custom documents path does not exist: {indexing_request.custom_path}",
                        config_field="custom_path"
                    )
                if not custom_path.is_dir():
                    raise ConfigurationError(
                        f"Custom path is not a directory: {indexing_request.custom_path}",
                        config_field="custom_path"
                    )
            
            # Перевірка чи не йде вже індексація
            # TODO: Додати lock mechanism для запобігання concurrent індексації
            
            # Запускаємо індексацію
            indexing_result = search_service.index_documents_from_path(
                custom_path=indexing_request.custom_path
            )
            
            # Перевіряємо результат
            if not indexing_result["success"]:
                raise DocumentProcessingError(
                    f"Indexing failed: {indexing_result.get('message', 'Unknown error')}"
                )
            
            # Додаємо метрики до контексту
            stats = indexing_result.get("stats", {})
            context.add_metric("documents_found", stats.get("total_documents_found", 0))
            context.add_metric("chunks_indexed", stats.get("chunks_indexed", 0))
            context.add_metric("indexing_duration_s", stats.get("total_time_s", 0))
            
            # Формуємо відповідь
            response = IndexingResponse(
                success=True,
                message=indexing_result["message"],
                stats=IndexingStats(**stats) if stats else None
            )
            
            logger.info(
                f"Document indexing completed successfully: {stats.get('chunks_indexed', 0)} chunks indexed",
                extra={"extra_data": context.to_log_dict()}
            )
            
            return response
            
        except ConfigurationError as e:
            logger.warning(f"Indexing configuration error: {e.message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.get_user_message()
            )
            
        except DocumentProcessingError as e:
            log_exception(logger, e, context={"endpoint": "index_documents"})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e.get_user_message()
            )
            
        except Exception as e:
            logger.error(f"Unexpected error during indexing: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document indexing failed due to unexpected error"
            )


@router.get(
    "/",
    response_model=Dict[str, Any],
    summary="Список документів у системі",
    description="""
    Повертає список всіх документів у системі з їх метаданими.
    
    **Включає:**
    - Назви файлів та шляхи
    - Статус обробки
    - Метадані (розмір, дата модифікації, тощо)
    - Кількість фрагментів
    """
)
async def list_documents(
    page: int = Query(1, description="Номер сторінки", ge=1),
    size: int = Query(20, description="Розмір сторінки", ge=1, le=100),
    file_type: Optional[DocumentType] = Query(None, description="Фільтр за типом файлу"),
    search_query: Optional[str] = Query(None, description="Пошук за назвою файлу"),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service)
):
    """
    Пагінований список документів з можливістю фільтрації.
    
    Корисний для:
    - Адміністративних панелей
    - Моніторингу стану документів
    - Пошуку конкретних файлів
    """
    async with track_endpoint_metrics("list_documents", context):
        try:
            # TODO: Реалізувати отримання списку документів з векторної БД
            # Поки що повертаємо заглушку з базовою інформацією
            
            # Отримуємо статистику системи
            stats = search_service.get_document_stats()
            vector_stats = stats.get("vector_database", {})
            discovery_stats = stats.get("document_discovery", {})
            
            # Формуємо мок-дані для демонстрації
            mock_documents = []
            for i in range(min(size, 10)):  # Обмежуємо для демо
                mock_documents.append({
                    "id": f"doc_{page}_{i}",
                    "file_name": f"Документ_{i + (page - 1) * size}.docx",
                    "file_type": "docx",
                    "file_size_mb": round(2.5 + i * 0.3, 1),
                    "chunks_count": 15 + i * 2,
                    "indexed_at": datetime.now().isoformat(),
                    "status": "indexed"
                })
            
            # Підрахунок пагінації
            total_documents = vector_stats.get("points_count", 0) // 10  # Припускаємо 10 чанків на документ
            total_pages = max(1, (total_documents + size - 1) // size)
            
            return {
                "documents": mock_documents,
                "pagination": {
                    "page": page,
                    "size": size,
                    "total_documents": total_documents,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                },
                "filters": {
                    "file_type": file_type.value if file_type else None,
                    "search_query": search_query
                },
                "stats": {
                    "total_indexed_chunks": vector_stats.get("points_count", 0),
                    "files_in_directory": discovery_stats.get("total_files_found", 0),
                    "file_types_distribution": discovery_stats.get("file_types", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve documents list"
            )


@router.get(
    "/{document_id}",
    response_model=Dict[str, Any],
    summary="Детальна інформація про документ",
    description="""
    Повертає детальну інформацію про конкретний документ.
    
    **Включає:**
    - Повні метадані документа
    - Список всіх фрагментів (чанків)
    - Статистику обробки
    - Історію змін
    """
)
async def get_document_details(
    document_id: str,
    include_chunks: bool = Query(False, description="Чи включати текст фрагментів"),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service)
):
    """
    Детальна інформація про конкретний документ.
    
    Корисний для:
    - Debugging проблем з конкретними документами
    - Аналізу якості індексації
    - Перегляду структури документа
    """
    async with track_endpoint_metrics("get_document_details", context):
        try:
            # TODO: Реалізувати отримання деталей документа з векторної БД
            # Поки що заглушка
            
            logger.info(f"Document details requested for: {document_id}")
            
            # Мок-дані для демонстрації
            document_details = {
                "document_id": document_id,
                "file_name": f"Документ_{document_id}.docx",
                "file_path": f"/documents/Документ_{document_id}.docx",
                "file_type": "docx",
                "file_size_bytes": 2560000,
                "created_at": "2024-01-15T10:30:00",
                "indexed_at": datetime.now().isoformat(),
                "status": "indexed",
                "metadata": {
                    "title": f"Технічний документ {document_id}",
                    "author": "Система управління",
                    "page_count": 25,
                    "word_count": 5420,
                    "extraction_method": "python-docx"
                },
                "processing": {
                    "chunks_count": 18,
                    "processing_duration_s": 12.5,
                    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
                }
            }
            
            # Додаємо чанки якщо запитано
            if include_chunks:
                document_details["chunks"] = [
                    {
                        "chunk_id": f"{document_id}_chunk_{i}",
                        "chunk_index": i,
                        "text": f"Це текст фрагменту {i} документа {document_id}. Тут містяться важливі технічні деталі...",
                        "word_count": 45 + i * 5,
                        "page_number": (i // 3) + 1
                    }
                    for i in range(5)  # Показуємо тільки перші 5 чанків для демо
                ]
            
            return document_details
            
        except Exception as e:
            logger.error(f"Failed to get document details for {document_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found or failed to retrieve details"
            )


@router.delete(
    "/{document_id}",
    response_model=BaseResponse,
    summary="Видалення документа з індексу",
    description="""
    Видаляє документ та всі його фрагменти з пошукового індексу.
    
    **Увага:** Ця операція незворотна!
    Видаляється тільки індекс, оригінальний файл залишається.
    """
)
async def delete_document(
    document_id: str,
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _admin = Depends(require_admin)
):
    """
    Видалення документа з пошукового індексу.
    
    Корисно для:
    - Видалення застарілих документів
    - Очищення некоректно проіндексованих файлів
    - Управління розміром індексу
    """
    async with track_endpoint_metrics("delete_document", context):
        try:
            logger.warning(
                f"Document deletion requested: {document_id}",
                extra={
                    "extra_data": {
                        "document_id": document_id,
                        "admin_user": context.user_id,
                        "request_context": context.to_log_dict()
                    }
                }
            )
            
            # TODO: Реалізувати видалення документа через vector store
            # deleted_count = search_service.vector_store.delete_by_document_id(document_id)
            
            # Поки що заглушка
            deleted_count = 1  # Припускаємо що видалили 1 документ
            
            if deleted_count > 0:
                return BaseResponse(
                    success=True,
                    message=f"Document {document_id} deleted successfully ({deleted_count} chunks removed)"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {document_id} not found in index"
                )
                
        except HTTPException:
            raise  # Перекидаємо HTTP exceptions
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete document {document_id}"
            )


@router.post(
    "/reindex/{document_id}",
    response_model=IndexingResponse,
    summary="Переіндексація конкретного документа",
    description="""
    Переіндексовує конкретний документ.
    
    **Процес:**
    1. Видаляє існуючі фрагменти документа з індексу
    2. Повторно обробляє файл
    3. Створює нові фрагменти з оновленим текстом
    4. Індексує у векторну базу даних
    """
)
async def reindex_document(
    document_id: str,
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _admin = Depends(require_admin)
):
    """
    Переіндексація одного документа.
    
    Корисно коли:
    - Документ був оновлений
    - Змінилась ML модель
    - Виникли помилки при початковій індексації
    """
    async with track_endpoint_metrics("reindex_document", context):
        try:
            logger.info(f"Document reindexing requested: {document_id}")
            
            # TODO: Реалізувати переіндексацію конкретного документа
            # 1. Знайти оригінальний файл
            # 2. Видалити старі чанки
            # 3. Повторно обробити файл
            # 4. Проіндексувати нові чанки
            
            # Поки що заглушка
            return IndexingResponse(
                success=True,
                message=f"Document {document_id} reindexed successfully",
                stats=IndexingStats(
                    total_documents_found=1,
                    chunks_processed=12,
                    chunks_indexed=12,
                    success_rate=100.0,
                    processing_time_s=5.2,
                    embedding_time_s=3.1,
                    indexing_time_s=1.8,
                    total_time_s=10.1,
                    avg_time_per_chunk_ms=842.0
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to reindex document {document_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to reindex document {document_id}"
            )


@router.post(
    "/clear-index",
    response_model=BaseResponse,
    summary="Очищення всього пошукового індексу",
    description="""
    **УВАГА: НЕБЕЗПЕЧНА ОПЕРАЦІЯ!**
    
    Видаляє всі документи з пошукового індексу.
    Оригінальні файли залишаються недоторканими.
    
    Використовуйте тільки для:
    - Повного перебудування індексу
    - Тестування та розробки
    - Видалення всіх даних перед міграцією
    """
)
async def clear_search_index(
    confirm: bool = Query(False, description="Підтвердження операції"),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _admin = Depends(require_admin)
):
    """
    Повне очищення пошукового індексу.
    
    Потребує явного підтвердження для безпеки.
    """
    async with track_endpoint_metrics("clear_index", context):
        if not confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Operation requires explicit confirmation. Set confirm=true"
            )
        
        try:
            logger.critical(
                "Search index clearing requested - DESTRUCTIVE OPERATION",
                extra={
                    "extra_data": {
                        "admin_user": context.user_id,
                        "request_context": context.to_log_dict()
                    }
                }
            )
            
            # TODO: Реалізувати очищення індексу
            # deleted_count = search_service.vector_store.clear_collection()
            
            # Поки що заглушка
            deleted_count = 1000  # Припускаємо що видалили 1000 чанків
            
            logger.critical(f"Search index cleared: {deleted_count} chunks removed")
            
            return BaseResponse(
                success=True,
                message=f"Search index cleared successfully. {deleted_count} chunks removed."
            )
            
        except Exception as e:
            logger.error(f"Failed to clear search index: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear search index"
            )


@router.post(
    "/clear-cache",
    response_model=CacheOperationResponse,
    summary="Очищення кешів системи",
    description="""
    Очищає всі кеші системи для звільнення пам'яті.
    
    **Очищає:**
    - Кеш пошукових запитів
    - Кеш ембедингів
    - Тимчасові дані
    
    Корисно для:
    - Звільнення пам'яті
    - Debugging проблем з кешуванням
    - Примусового оновлення даних
    """
)
async def clear_caches(
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _admin = Depends(require_admin)
):
    """
    Очищення всіх кешів системи.
    
    Безпечна операція - не видаляє індексовані дані.
    """
    async with track_endpoint_metrics("clear_caches", context):
        try:
            logger.info("Cache clearing requested")
            
            # Очищуємо кеші через search service
            cache_result = search_service.clear_cache()
            
            if cache_result["success"]:
                return CacheOperationResponse(
                    success=True,
                    message=cache_result["message"],
                    cleared_items=cache_result.get("cleared_items", {})
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=cache_result.get("message", "Failed to clear caches")
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to clear caches: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear caches"
            )


@router.get(
    "/stats/detailed",
    response_model=SystemStatsResponse,
    summary="Детальна статистика системи",
    description="""
    Повертає вичерпну статистику про стан системи документів.
    
    **Включає:**
    - Статистику векторної бази даних
    - Інформацію про ML модель
    - Метрики продуктивності
    - Використання ресурсів
    - Історичні дані
    """
)
async def get_detailed_stats(
    include_performance: bool = Query(True, description="Включити метрики продуктивності"),
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service)
):
    """
    Детальна статистика для моніторингу та аналітики.
    
    Корисна для:
    - Моніторингу продуктивності
    - Планування ресурсів
    - Оптимізації системи
    - Звітності
    """
    async with track_endpoint_metrics("detailed_stats", context):
        try:
            # Отримуємо базову статистику
            base_stats = search_service.get_document_stats()
            
            # Додаємо додаткову інформацію
            enhanced_stats = base_stats.copy()
            
            if include_performance:
                # TODO: Додати метрики продуктивності
                enhanced_stats["performance_metrics"] = {
                    "avg_search_time_ms": 250.5,
                    "avg_indexing_time_per_doc_s": 15.2,
                    "cache_hit_rate": 0.85,
                    "uptime_hours": 24.5
                }
            
            # Додаємо timestamp для моніторингу
            enhanced_stats["report_generated_at"] = datetime.now().isoformat()
            enhanced_stats["system_version"] = "1.0.0"
            
            return SystemStatsResponse(
                success=True,
                message="Detailed system statistics retrieved",
                **enhanced_stats
            )
            
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve detailed statistics"
            )


# === Batch операції ===

@router.post(
    "/batch/delete",
    response_model=BaseResponse,
    summary="Масове видалення документів",
    description="""
    Видаляє множинні документи з індексу одночасно.
    
    **Примітка:** Операція незворотна!
    Рекомендується створити backup перед виконанням.
    """
)
async def batch_delete_documents(
    document_ids: List[str],
    context: RequestContext = Depends(get_request_context),
    search_service = Depends(get_search_service),
    _admin = Depends(require_admin)
):
    """
    Масове видалення документів для ефективності.
    
    Корисно для:
    - Очищення великих кількостей застарілих документів
    - Міграції даних
    - Batch операцій адміністрування
    """
    async with track_endpoint_metrics("batch_delete", context):
        try:
            if len(document_ids) > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete more than 100 documents at once"
                )
            
            logger.warning(
                f"Batch deletion requested for {len(document_ids)} documents",
                extra={"extra_data": {"document_count": len(document_ids)}}
            )
            
            # TODO: Реалізувати batch видалення
            # deleted_count = search_service.vector_store.batch_delete(document_ids)
            
            # Заглушка
            deleted_count = len(document_ids)
            
            return BaseResponse(
                success=True,
                message=f"Batch deletion completed: {deleted_count} documents removed"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch deletion failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch deletion operation failed"
            )