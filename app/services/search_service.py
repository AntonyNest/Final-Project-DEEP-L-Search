# Search orchestration

"""
Search Service - Високорівнева оркестрація семантичного пошуку.

Цей сервіс реалізує pattern Facade та Command для координації всіх компонентів
системи пошуку. Він об'єднує:

1. Document Processing - для обробки нових документів
2. Embedding Service - для векторизації запитів та документів  
3. Vector Store - для зберігання та пошуку векторів
4. Result Processing - для пост-обробки та ранжування результатів

Архітектурні принципи:
- Single Responsibility: кожен метод відповідає за одну бізнес-операцію
- Dependency Injection: всі залежності передаються ззовні
- Error Recovery: graceful degradation при помилках компонентів
- Performance Optimization: кешування та batch операції
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time

from app.config import settings
from app.services.document_processor import DocumentProcessor, DocumentChunk
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store, SearchResult

logger = logging.getLogger(__name__)


class SearchStats:
    """
    Клас для збору статистики пошукових операцій.
    
    Ця інформація критично важлива для:
    1. Моніторингу продуктивності системи
    2. Виявлення bottlenecks в ML pipeline
    3. Оптимізації налаштувань системи
    4. A/B тестування різних підходів
    """
    
    def __init__(self):
        self.embedding_time: float = 0.0
        self.search_time: float = 0.0
        self.post_processing_time: float = 0.0
        self.total_time: float = 0.0
        self.results_count: int = 0
        self.query_length: int = 0
        self.timestamp: str = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник для логування та API відповідей."""
        return {
            "embedding_time_ms": round(self.embedding_time * 1000, 2),
            "search_time_ms": round(self.search_time * 1000, 2),
            "post_processing_time_ms": round(self.post_processing_time * 1000, 2),
            "total_time_ms": round(self.total_time * 1000, 2),
            "results_count": self.results_count,
            "query_length": self.query_length,
            "timestamp": self.timestamp
        }


class SearchService:
    """
    Головний сервіс для семантичного пошуку документів.
    
    Цей клас реалізує Command Pattern - кожен публічний метод представляє
    окрему бізнес-команду, яку може виконати система.
    """
    
    def __init__(self):
        """
        Ініціалізація з dependency injection pattern.
        
        Всі залежності ініціалізуються через глобальні екземпляри,
        що дозволяє легко замінити їх для тестування (mock objects).
        """
        self.document_processor = DocumentProcessor()
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
        # Кеш для часто використовуваних запитів
        self.query_cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        self.cache_ttl = 300  # 5 хвилин TTL для кешу
        
        logger.info("SearchService initialized with all dependencies")
    
    async def initialize_system(self) -> bool:
        """
        Ініціалізація всієї системи пошуку.
        
        Цей метод забезпечує правильну послідовність ініціалізації:
        1. Перевірка підключення до векторної БД
        2. Створення необхідних колекцій
        3. Перевірка доступності ML моделей
        4. Валідація конфігурації
        """
        try:
            logger.info("Initializing search system...")
            
            # Перевірка векторної БД та створення колекції
            if not await self.vector_store.ensure_collection_exists():
                logger.error("Failed to initialize vector store collection")
                return False
            
            # Перевірка доступності embedding service
            model_info = self.embedding_service.get_model_info()
            if not model_info.get('loaded', False):
                # Примусово завантажуємо модель для перевірки
                test_embedding = self.embedding_service.encode_single("test initialization")
                if test_embedding is None or len(test_embedding) == 0:
                    logger.error("Failed to initialize embedding service")
                    return False
            
            logger.info("Search system initialized successfully")
            logger.info(f"Model info: {model_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search system: {str(e)}")
            return False
    
    def index_documents_from_path(self, custom_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Повна індексація документів з заданого шляху.
        
        Цей метод реалізує ETL pipeline для документів:
        Extract -> Transform -> Load pattern
        
        Args:
            custom_path: Опціональний шлях до документів (за замовчуванням з config)
            
        Returns:
            Dict з результатами індексації та статистикою
        """
        start_time = time.time()
        
        try:
            # Використовуємо custom path якщо надано
            if custom_path:
                original_path = self.document_processor.documents_path
                self.document_processor.documents_path = Path(custom_path)
            
            logger.info(f"Starting document indexing from: {self.document_processor.documents_path}")
            
            # Етап 1: Виявлення та обробка документів
            processing_start = time.time()
            all_chunks = self.document_processor.process_all_documents()
            processing_time = time.time() - processing_start
            
            if not all_chunks:
                logger.warning("No document chunks were processed")
                return {
                    "success": False,
                    "message": "No documents found or processed",
                    "stats": {"processing_time": processing_time}
                }
            
            logger.info(f"Processed {len(all_chunks)} chunks in {processing_time:.2f}s")
            
            # Етап 2: Генерація ембедингів
            embedding_start = time.time()
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self.embedding_service.encode_batch(
                texts, 
                batch_size=32,  # Оптимальний розмір для більшості GPU
                use_cache=True
            )
            embedding_time = time.time() - embedding_start
            
            logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
            
            # Етап 3: Індексація у векторну БД
            indexing_start = time.time()
            indexed_count = self.vector_store.index_document_chunks_batch(
                chunks=all_chunks,
                embeddings=embeddings,
                batch_size=100  # Більший batch для векторної БД
            )
            indexing_time = time.time() - indexing_start
            
            total_time = time.time() - start_time
            
            # Статистика індексації
            stats = {
                "total_documents_found": len(self.document_processor.discover_documents()),
                "chunks_processed": len(all_chunks),
                "chunks_indexed": indexed_count,
                "success_rate": round((indexed_count / len(all_chunks)) * 100, 2) if all_chunks else 0,
                "processing_time_s": round(processing_time, 2),
                "embedding_time_s": round(embedding_time, 2),
                "indexing_time_s": round(indexing_time, 2),
                "total_time_s": round(total_time, 2),
                "avg_time_per_chunk_ms": round((total_time / len(all_chunks)) * 1000, 2) if all_chunks else 0
            }
            
            # Відновлюємо оригінальний шлях якщо змінювали
            if custom_path:
                self.document_processor.documents_path = original_path
            
            success = indexed_count > 0
            message = f"Successfully indexed {indexed_count}/{len(all_chunks)} chunks"
            
            if not success:
                message = "Failed to index any chunks"
            elif indexed_count < len(all_chunks):
                message += f" (warning: {len(all_chunks) - indexed_count} chunks failed)"
            
            logger.info(f"Indexing completed: {message}")
            
            return {
                "success": success,
                "message": message,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {str(e)}")
            return {
                "success": False,
                "message": f"Indexing failed: {str(e)}",
                "stats": {"total_time_s": time.time() - start_time}
            }
    
    def search(
        self,
        query: str,
        limit: int = None,
        score_threshold: float = None,
        filters: Optional[Dict[str, Any]] = None,
        include_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Основний метод семантичного пошуку.
        
        Цей метод реалізує повний pipeline пошуку:
        1. Валідація та нормалізація запиту
        2. Перевірка кеша
        3. Генерація ембединга запиту
        4. Векторний пошук
        5. Пост-обробка результатів
        6. Кешування результатів
        
        Args:
            query: Текстовий запит для пошуку
            limit: Максимальна кількість результатів
            score_threshold: Мінімальний поріг схожості
            filters: Фільтри за метаданими документів
            include_stats: Чи включати статистику виконання
            
        Returns:
            Dict з результатами пошуку та опціональною статистикою
        """
        start_time = time.time()
        stats = SearchStats()
        stats.query_length = len(query)
        
        try:
            # Валідація запиту
            if not query or not query.strip():
                return {
                    "success": False,
                    "message": "Empty query provided",
                    "results": [],
                    "stats": stats.to_dict() if include_stats else None
                }
            
            normalized_query = query.strip()
            
            # Перевірка кеша (тільки для запитів без фільтрів)
            cache_key = None
            if not filters:
                cache_key = f"{normalized_query}:{limit}:{score_threshold}"
                if cache_key in self.query_cache:
                    cached_results, cache_time = self.query_cache[cache_key]
                    # Перевіряємо TTL кеша
                    if time.time() - cache_time < self.cache_ttl:
                        logger.debug(f"Cache hit for query: {normalized_query[:50]}...")
                        stats.total_time = time.time() - start_time
                        return {
                            "success": True,
                            "query": normalized_query,
                            "results": [result.__dict__ for result in cached_results],
                            "cached": True,
                            "stats": stats.to_dict() if include_stats else None
                        }
            
            # Генерація ембединга для запиту
            embedding_start = time.time()
            query_embedding = self.embedding_service.encode_single(normalized_query)
            stats.embedding_time = time.time() - embedding_start
            
            if query_embedding is None or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for query")
                return {
                    "success": False,
                    "message": "Failed to process query",
                    "results": [],
                    "stats": stats.to_dict() if include_stats else None
                }
            
            # Векторний пошук
            search_start = time.time()
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters
            )
            stats.search_time = time.time() - search_start
            
            # Пост-обробка результатів
            processing_start = time.time()
            processed_results = self._post_process_results(search_results, normalized_query)
            stats.post_processing_time = time.time() - processing_start
            stats.results_count = len(processed_results)
            
            # Кешування результатів (тільки для запитів без фільтрів)
            if cache_key and processed_results:
                self.query_cache[cache_key] = (processed_results, time.time())
                # Обмежуємо розмір кеша
                if len(self.query_cache) > 100:
                    # Видаляємо найстаріші записи
                    oldest_key = min(self.query_cache.keys(), 
                                   key=lambda k: self.query_cache[k][1])
                    del self.query_cache[oldest_key]
            
            stats.total_time = time.time() - start_time
            
            logger.info(
                f"Search completed: query='{normalized_query[:50]}...', "
                f"results={stats.results_count}, time={stats.total_time:.3f}s"
            )
            
            return {
                "success": True,
                "query": normalized_query,
                "results": [result.__dict__ for result in processed_results],
                "cached": False,
                "stats": stats.to_dict() if include_stats else None
            }
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            stats.total_time = time.time() - start_time
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "results": [],
                "stats": stats.to_dict() if include_stats else None
            }
    
    def _post_process_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Пост-обробка результатів пошуку для покращення релевантності.
        
        Цей метод реалізує додаткові алгоритми ранжування поверх
        векторної схожості. Включає:
        
        1. Keyword Boost - підвищення рейтингу при точному співпадінні слів
        2. Document Diversity - уникнення дублікатів з одного файлу
        3. Length Normalization - нормалізація за довжиною тексту
        4. Recency Boost - підвищення рейтингу новіших документів
        """
        if not results:
            return results
        
        query_words = set(query.lower().split())
        processed_results = []
        
        for result in results:
            # Створюємо копію для модифікації
            enhanced_result = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                source_file=result.source_file,
                metadata=result.metadata.copy()
            )
            
            # Keyword Boost: підвищуємо score при точному співпадінні слів
            text_words = set(result.text.lower().split())
            common_words = query_words.intersection(text_words)
            if common_words:
                keyword_boost = min(0.1, len(common_words) / len(query_words) * 0.1)
                enhanced_result.score = min(1.0, enhanced_result.score + keyword_boost)
                enhanced_result.metadata['keyword_matches'] = list(common_words)
                enhanced_result.metadata['keyword_boost'] = keyword_boost
            
            # Length Normalization: штрафуємо занадто короткі або довгі тексти
            text_length = len(result.text.split())
            if text_length < 10:  # Занадто короткий текст
                enhanced_result.score *= 0.9
            elif text_length > 500:  # Занадто довгий текст
                enhanced_result.score *= 0.95
            
            # Додаємо метадані для debugging
            enhanced_result.metadata['original_score'] = result.score
            enhanced_result.metadata['text_length_words'] = text_length
            
            processed_results.append(enhanced_result)
        
        # Document Diversity: групуємо за файлами та обмежуємо кількість результатів з одного файлу
        file_counts = {}
        diverse_results = []
        max_per_file = max(1, len(results) // 3)  # Максимум 1/3 результатів з одного файлу
        
        for result in sorted(processed_results, key=lambda r: r.score, reverse=True):
            file_key = result.source_file
            current_count = file_counts.get(file_key, 0)
            
            if current_count < max_per_file:
                diverse_results.append(result)
                file_counts[file_key] = current_count + 1
            else:
                # Додаємо до кінця списку з пониженим score для різноманітності
                result.score *= 0.8
                result.metadata['diversity_penalty'] = True
                diverse_results.append(result)
        
        # Повторно сортуємо за фінальним score
        diverse_results.sort(key=lambda r: r.score, reverse=True)
        
        return diverse_results
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Отримання статистики про проіндексовані документи.
        
        Ця інформація допомагає зрозуміти:
        1. Розмір корпусу документів
        2. Розподіл типів файлів
        3. Якість індексації
        4. Продуктивність системи
        """
        try:
            # Статистика з векторної БД
            vector_stats = self.vector_store.get_collection_stats()
            
            # Статистика з файлової системи
            documents = self.document_processor.discover_documents()
            file_types = {}
            total_size = 0
            
            for doc_path in documents:
                ext = doc_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                total_size += doc_path.stat().st_size
            
            # Статистика моделі
            model_info = self.embedding_service.get_model_info()
            
            combined_stats = {
                "document_discovery": {
                    "total_files_found": len(documents),
                    "file_types": file_types,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                },
                "vector_database": vector_stats,
                "embedding_model": model_info,
                "system_health": {
                    "vector_db_healthy": self.vector_store.health_check(),
                    "model_loaded": model_info.get('loaded', False),
                    "cache_size": len(self.query_cache)
                }
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Очищення всіх кешів системи.
        
        Корисно для:
        1. Звільнення пам'яті
        2. Перевірки актуальності результатів
        3. Debugging проблем з кешуванням
        """
        try:
            # Очищення кеша запитів
            query_cache_size = len(self.query_cache)
            self.query_cache.clear()
            
            # Очищення кеша ембедингів
            embedding_cache_size = len(self.embedding_service.cache.memory_cache)
            self.embedding_service.cache.memory_cache.clear()
            
            logger.info(
                f"Caches cleared: query_cache={query_cache_size}, "
                f"embedding_cache={embedding_cache_size}"
            )
            
            return {
                "success": True,
                "message": "All caches cleared successfully",
                "cleared_items": {
                    "query_cache": query_cache_size,
                    "embedding_cache": embedding_cache_size
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to clear caches: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to clear caches: {str(e)}"
            }


# Глобальний екземпляр сервісу
search_service = SearchService()