# Qdrant integration

"""
Vector Store Service - Управління векторною базою даних Qdrant.

Цей сервіс реалізує шаблон Repository Pattern для роботи з векторними даними.
Він абстрагує всі деталі взаємодії з Qdrant та надає простий API для:

1. Створення та управління колекціями (collections)
2. Індексації векторів з метаданими
3. Семантичного пошуку з фільтрацією
4. Batch операцій для ефективності

Ключові концепції векторних БД:
- Collection: група векторів з однаковими характеристиками
- Point: окремий вектор з унікальним ID та метаданими  
- Index: структура даних для швидкого пошуку найближчих векторів
- Filter: умови для фільтрації результатів за метаданими
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionStatus,
    PointStruct, SearchRequest, Filter,
    FieldCondition, MatchValue, Range,
    UpdateResult, ScrollRequest
)
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.services.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Структура для результатів пошуку.
    
    Ця модель інкапсулює всю необхідну інформацію про знайдений документ,
    забезпечуючи type safety та зручність використання в API.
    """
    chunk_id: str
    text: str
    score: float  # Similarity score (0.0 - 1.0)
    source_file: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Валідація та нормалізація даних після ініціалізації."""
        # Обмежуємо score в допустимих межах
        self.score = max(0.0, min(1.0, self.score))
        
        # Забезпечуємо наявність базових метаданих
        if 'file_name' not in self.metadata and self.source_file:
            self.metadata['file_name'] = self.source_file.split('/')[-1]


class VectorStoreService:
    """
    Основний сервіс для роботи з векторною базою даних Qdrant.
    
    Архітектурні принципи:
    1. Connection Pooling - ефективне управління з'єднаннями
    2. Error Recovery - graceful handling помилок мережі та БД
    3. Batch Operations - оптимізація для великих обсягів даних
    4. Schema Validation - забезпечення цілісності даних
    """
    
    def __init__(self):
        """Ініціалізація сервісу з конфігурацією підключення."""
        self.client = None
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.embedding_dimension
        self.host = settings.qdrant_host
        self.port = settings.qdrant_port
        
        logger.info(
            f"VectorStoreService initialized for collection '{self.collection_name}' "
            f"at {self.host}:{self.port}"
        )
    
    def _get_client(self) -> QdrantClient:
        """
        Lazy initialization клієнта Qdrant з connection pooling.
        
        Використовуємо pattern Lazy Loading для уникнення проблем
        з підключенням при старті сервісу, коли Qdrant може бути ще недоступний.
        """
        if self.client is None:
            try:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=30.0,  # Таймаут для важких операцій
                    # Connection pooling для кращої продуктивності
                    prefer_grpc=True,  # gRPC швидший за HTTP для bulk операцій
                )
                
                # Перевіряємо з'єднання
                health = self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant. Collections: {len(health.collections)}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                raise ConnectionError(f"Could not establish connection to Qdrant: {str(e)}")
        
        return self.client
    
    async def ensure_collection_exists(self) -> bool:
        """
        Створення колекції, якщо вона не існує.
        
        Колекція в Qdrant - це логічна група векторів з однаковими характеристиками.
        Налаштовуємо оптимальні параметри для семантичного пошуку документів.
        """
        try:
            client = self._get_client()
            
            # Перевіряємо, чи існує колекція
            try:
                collection_info = client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
                
                # Валідуємо параметри існуючої колекції
                vector_config = collection_info.config.params.vectors
                if vector_config.size != self.embedding_dimension:
                    logger.error(
                        f"Dimension mismatch: collection has {vector_config.size}, "
                        f"expected {self.embedding_dimension}"
                    )
                    return False
                
                return True
                
            except UnexpectedResponse as e:
                if "not found" in str(e).lower():
                    # Колекція не існує, створюємо нову
                    logger.info(f"Creating new collection '{self.collection_name}'")
                    
                    # Налаштування векторних параметрів
                    # Cosine distance оптимальна для текстових ембедингів
                    vector_params = VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,  # Косинусна відстань для семантичної схожості
                    )
                    
                    # Створюємо колекцію з оптимізованими налаштуваннями
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=vector_params,
                        # Оптимізація для швидкого пошуку
                        optimizers_config={
                            "default_segment_number": 2,  # Кількість сегментів для індексу
                            "max_segment_size": 20000,    # Максимальний розмір сегменту
                        },
                        # Налаштування реплікації для production
                        replication_factor=1,  # В production варто збільшити до 2-3
                        write_consistency_factor=1,
                    )
                    
                    logger.info(f"Collection '{self.collection_name}' created successfully")
                    return True
                else:
                    raise e
                    
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            return False
    
    def index_document_chunk(self, chunk: DocumentChunk, embedding: np.ndarray) -> bool:
        """
        Індексація одного чанку документа з його ембедингом.
        
        Args:
            chunk: Об'єкт DocumentChunk з текстом та метаданими
            embedding: Векторне представлення чанку
            
        Returns:
            bool: True якщо індексація успішна
        """
        try:
            client = self._get_client()
            
            # Підготовка метаданих для зберігання
            # Важливо: Qdrant зберігає метадані окремо від векторів
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "created_at": chunk.created_at,
                **chunk.metadata  # Розпаковуємо всі додаткові метадані
            }
            
            # Генеруємо унікальний ID для point в Qdrant
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            # Створюємо Point object для Qdrant
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),  # Qdrant очікує list, не numpy array
                payload=payload
            )
            
            # Вставляємо point в колекцію
            result = client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True  # Чекаємо підтвердження запису
            )
            
            if result.status == "completed":
                logger.debug(f"Successfully indexed chunk: {chunk.chunk_id}")
                return True
            else:
                logger.warning(f"Unexpected indexing result: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to index chunk {chunk.chunk_id}: {str(e)}")
            return False
    
    def index_document_chunks_batch(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[np.ndarray],
        batch_size: int = 100
    ) -> int:
        """
        Батчева індексація множинних чанків для ефективності.
        
        Батчева обробка критично важлива для великих документів,
        оскільки зменшує кількість мережевих викликів та використовує
        bulk операції Qdrant для максимальної продуктивності.
        
        Args:
            chunks: Список чанків для індексації
            embeddings: Відповідні ембединги для кожного чанку  
            batch_size: Розмір батчу (компроміс між швидкістю та пам'яттю)
            
        Returns:
            int: Кількість успішно проіндексованих чанків
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        indexed_count = 0
        total_chunks = len(chunks)
        
        logger.info(f"Starting batch indexing of {total_chunks} chunks (batch_size={batch_size})")
        
        try:
            client = self._get_client()
            
            # Обробляємо чанки батчами
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                # Підготовляємо Points для батчу
                points = []
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    # Підготовка payload
                    payload = {
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "word_count": chunk.word_count,
                        "char_count": chunk.char_count,
                        "created_at": chunk.created_at,
                        **chunk.metadata
                    }
                    
                    # Генеруємо унікальний ID
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                    points.append(point)
                
                # Виконуємо батчеву вставку
                try:
                    result = client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    
                    if result.status == "completed":
                        batch_indexed = len(points)
                        indexed_count += batch_indexed
                        logger.info(
                            f"Batch {i//batch_size + 1}: indexed {batch_indexed} chunks "
                            f"({indexed_count}/{total_chunks} total)"
                        )
                    else:
                        logger.warning(f"Batch indexing returned unexpected status: {result.status}")
                        
                except Exception as batch_error:
                    logger.error(f"Failed to index batch {i//batch_size + 1}: {str(batch_error)}")
                    # Продовжуємо з наступним батчем
                    continue
            
            logger.info(f"Batch indexing completed: {indexed_count}/{total_chunks} chunks indexed")
            return indexed_count
            
        except Exception as e:
            logger.error(f"Failed during batch indexing: {str(e)}")
            return indexed_count
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = None,
        score_threshold: float = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Семантичний пошук схожих документів за ембедингом запиту.
        
        Це основний метод для семантичного пошуку, який використовує
        векторну схожість для знаходження релевантних документів.
        
        Args:
            query_embedding: Векторне представлення пошукового запиту
            limit: Максимальна кількість результатів (за замовчуванням з config)
            score_threshold: Мінімальний поріг схожості (за замовчуванням з config)
            filters: Фільтри за метаданими (file_type, source_file, тощо)
            
        Returns:
            List[SearchResult]: Відсортований список результатів за релевантністю
        """
        # Використовуємо значення за замовчуванням з конфігурації
        limit = limit or settings.default_limit
        score_threshold = score_threshold or settings.similarity_threshold
        
        try:
            client = self._get_client()
            
            # Підготовка фільтрів для Qdrant
            qdrant_filter = None
            if filters:
                conditions = []
                
                for field, value in filters.items():
                    if isinstance(value, str):
                        # Точне співпадіння для рядків
                        conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )
                    elif isinstance(value, (int, float)):
                        # Для чисел також точне співпадіння
                        conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )
                    elif isinstance(value, dict) and 'range' in value:
                        # Діапазонний фільтр для чисел
                        range_filter = value['range']
                        conditions.append(
                            FieldCondition(
                                key=field,
                                range=Range(
                                    gte=range_filter.get('gte'),
                                    lte=range_filter.get('lte')
                                )
                            )
                        )
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Виконуємо векторний пошук
            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,  # Включаємо метадані в результати
                with_vectors=False  # Не повертаємо вектори для економії bandwidth
            )
            
            # Конвертуємо результати в наш формат
            results = []
            for hit in search_results:
                # Витягуємо дані з payload
                payload = hit.payload
                
                search_result = SearchResult(
                    chunk_id=payload.get('chunk_id', 'unknown'),
                    text=payload.get('text', ''),
                    score=hit.score,
                    source_file=payload.get('source_file', ''),
                    metadata={
                        k: v for k, v in payload.items() 
                        if k not in ['text', 'chunk_id', 'source_file']
                    }
                )
                results.append(search_result)
            
            logger.info(
                f"Search completed: found {len(results)} results "
                f"(threshold: {score_threshold}, limit: {limit})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def search_by_text(
        self,
        query_text: str,
        embedding_service,  # Щоб уникнути circular import
        limit: int = None,
        score_threshold: float = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Зручний метод для пошуку за текстовим запитом.
        
        Цей метод об'єднує генерацію ембединга та пошук в одному API call,
        що спрощує використання сервісу для кінцевих користувачів.
        """
        try:
            # Генеруємо ембединг для запиту
            query_embedding = embedding_service.encode_single(query_text)
            
            # Виконуємо пошук
            return self.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Text search failed for query '{query_text}': {str(e)}")
            return []
    
    def delete_by_source_file(self, source_file: str) -> int:
        """
        Видалення всіх чанків з конкретного файлу.
        
        Корисно для оновлення документів - спочатку видаляємо старі чанки,
        потім індексуємо нові версії документа.
        """
        try:
            client = self._get_client()
            
            # Створюємо фільтр для знаходження всіх чанків файлу
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file)
                    )
                ]
            )
            
            # Виконуємо видалення
            result = client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter,
                wait=True
            )
            
            deleted_count = result.operation_id  # Кількість видалених points
            logger.info(f"Deleted {deleted_count} chunks from file: {source_file}")
            
            return deleted_count if deleted_count else 0
            
        except Exception as e:
            logger.error(f"Failed to delete chunks from {source_file}: {str(e)}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Отримання статистики колекції для моніторингу.
        
        Ця інформація корисна для:
        1. Моніторингу здоров'я системи
        2. Планування масштабування
        3. Debugging проблем з індексацією
        """
        try:
            client = self._get_client()
            
            # Отримуємо інформацію про колекцію
            collection_info = client.get_collection(self.collection_name)
            
            # Отримуємо статистику points
            stats = {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Перевірка здоров'я векторної бази даних.
        
        Використовується для health endpoints API та моніторингу системи.
        """
        try:
            client = self._get_client()
            
            # Простий запит для перевірки доступності
            collections = client.get_collections()
            
            # Перевіряємо, чи існує наша колекція
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# Глобальний екземпляр сервісу
vector_store = VectorStoreService()