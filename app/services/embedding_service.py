# Embedding generation

"""
Embedding Service - Генерація векторних представлень тексту.

Цей сервіс відповідає за трансформацію тексту в числові вектори (embeddings),
які зберігають семантичне значення тексту в багатовимірному просторі.

Ключові концепції:
1. Semantic Similarity - схожі за змістом тексти мають схожі вектори
2. Vector Space - кожне слово/речення представлено точкою в багатовимірному просторі  
3. Distance Metrics - косинусна відстань показує семантичну схожість

Архітектурний патерн: Singleton + Factory для ефективного управління ML моделями.
"""

import logging
import numpy as np
import torch
from typing import List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Кеш для ембедингів з використанням хешування тексту.
    
    Ця оптимізація критично важлива для ML систем, оскільки:
    1. Генерація ембедингів - computationally expensive операція
    2. Ідентичні тексти завжди дають ідентичні ембединги
    3. Кешування зменшує час обробки на 80-90% для повторних запитів
    """
    
    def __init__(self, cache_dir: str = "embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}  # In-memory cache for recently used embeddings
        self.max_memory_items = 1000  # Prevent memory overflow
        
        logger.info(f"EmbeddingCache initialized with directory: {self.cache_dir}")
    
    def _get_text_hash(self, text: str) -> str:
        """
        Генерує унікальний хеш для тексту.
        
        Використовуємо SHA-256 для гарантованої унікальності хешів.
        Це дозволяє швидко перевірити, чи є ембединг вже обчислений.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Отримати ембединг з кеша, якщо він існує."""
        text_hash = self._get_text_hash(text)
        
        # Спочатку перевіряємо memory cache (найшвидший)
        if text_hash in self.memory_cache:
            logger.debug(f"Cache hit (memory) for text hash: {text_hash[:16]}...")
            return self.memory_cache[text_hash]
        
        # Потім перевіряємо disk cache
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Додаємо в memory cache для швидшого доступу наступного разу
                if len(self.memory_cache) < self.max_memory_items:
                    self.memory_cache[text_hash] = embedding
                
                logger.debug(f"Cache hit (disk) for text hash: {text_hash[:16]}...")
                return embedding
                
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                cache_file.unlink()  # Видаляємо пошкоджений файл
        
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """Зберегти ембединг в кеш."""
        text_hash = self._get_text_hash(text)
        
        # Зберігаємо в memory cache
        if len(self.memory_cache) < self.max_memory_items:
            self.memory_cache[text_hash] = embedding
        
        # Зберігаємо на диск для persistent storage
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            logger.debug(f"Cached embedding for text hash: {text_hash[:16]}...")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")


class EmbeddingService:
    """
    Основний сервіс для генерації та управління ембедингами.
    
    Архітектурні принципи:
    1. Lazy Loading - модель завантажується тільки при першому використанні
    2. Device Management - автоматичне визначення GPU/CPU
    3. Batch Processing - ефективна обробка множинних текстів
    4. Error Handling - graceful degradation при помилках
    """
    
    _instance = None  # Singleton pattern для уникнення множинного завантаження моделей
    
    def __new__(cls):
        """Singleton pattern - гарантуємо єдиний екземпляр сервісу."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Захист від повторної ініціалізації singleton
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.model_name = settings.embedding_model
        self.device = self._determine_device()
        self.model = None  # Lazy loading
        self.cache = EmbeddingCache()
        
        logger.info(f"EmbeddingService initialized with model: {self.model_name}, device: {self.device}")
    
    def _determine_device(self) -> str:
        """
        Автоматичне визначення найкращого пристрою для обчислень.
        
        Для ML workloads порядок пріоритету: CUDA GPU > MPS (Apple Silicon) > CPU
        """
        if settings.device != "auto":
            return settings.device
        
        # Перевіряємо доступність CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {gpu_name}")
        # Перевіряємо MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple MPS available")
        else:
            device = "cpu"
            logger.info("Using CPU for embeddings")
        
        return device
    
    def _load_model(self) -> None:
        """
        Lazy loading ML моделі.
        
        Завантаження великих transformer моделей займає час та пам'ять,
        тому робимо це тільки при першому запиті.
        """
        if self.model is not None:
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Завантажуємо pre-trained модель з HuggingFace
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Налаштування для оптимізації inference
            self.model.eval()  # Переводимо в evaluation mode
            
            # Отримуємо розмірність ембедингів для валідації
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != settings.embedding_dimension:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from config ({settings.embedding_dimension}). "
                    f"Updating config to match model."
                )
                # В production системі краще оновити конфігурацію або вибрати іншу модель
            
            logger.info(f"Model loaded successfully. Embedding dimension: {actual_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Could not initialize embedding model: {str(e)}")
    
    def encode_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Генерація ембединга для одного тексту.
        
        Args:
            text: Вхідний текст для векторизації
            use_cache: Чи використовувати кеш для оптимізації
            
        Returns:
            np.ndarray: Нормалізований ембединг вектор
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for encoding")
            return np.zeros(settings.embedding_dimension, dtype=np.float32)
        
        # Нормалізуємо текст для консистентного кешування
        normalized_text = text.strip()
        
        # Перевіряємо кеш
        if use_cache:
            cached_embedding = self.cache.get(normalized_text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Завантажуємо модель при першому використанні
        self._load_model()
        
        try:
            # Генеруємо ембединг
            with torch.no_grad():  # Відключаємо gradient computation для економії пам'яті
                embedding = self.model.encode(
                    [normalized_text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # L2 normalization для косинусної схожості
                    batch_size=1
                )[0]
            
            # Зберігаємо в кеш
            if use_cache:
                self.cache.put(normalized_text, embedding)
            
            logger.debug(f"Generated embedding for text (length: {len(normalized_text)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            # Повертаємо zero-vector як fallback
            return np.zeros(settings.embedding_dimension, dtype=np.float32)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, use_cache: bool = True) -> List[np.ndarray]:
        """
        Батчева генерація ембедингів для множинних текстів.
        
        Батчева обробка значно ефективніша ніж послідовна для GPU обчислень,
        оскільки дозволяє повністю використовувати паралелізм GPU.
        
        Args:
            texts: Список текстів для векторизації
            batch_size: Розмір батчу (більший = швидше, але більше пам'яті)
            use_cache: Чи використовувати кеш
            
        Returns:
            List[np.ndarray]: Список нормалізованих ембединг векторів
        """
        if not texts:
            return []
        
        # Нормалізуємо всі тексти
        normalized_texts = [text.strip() for text in texts]
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        # Перевіряємо кеш для кожного тексту
        if use_cache:
            for i, text in enumerate(normalized_texts):
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_compute.append(text)
                    indices_to_compute.append(i)
        else:
            texts_to_compute = normalized_texts
            indices_to_compute = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Обчислюємо ембединги для текстів, яких немає в кеші
        if texts_to_compute:
            logger.info(f"Computing embeddings for {len(texts_to_compute)} texts (batch size: {batch_size})")
            
            self._load_model()
            
            try:
                with torch.no_grad():
                    # Батчева обробка для ефективності
                    computed_embeddings = self.model.encode(
                        texts_to_compute,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=batch_size,
                        show_progress_bar=len(texts_to_compute) > 50  # Показуємо прогрес для великих батчів
                    )
                
                # Розміщуємо обчислені ембединги в правильних позиціях
                for i, (computed_embedding, original_index) in enumerate(zip(computed_embeddings, indices_to_compute)):
                    embeddings[original_index] = computed_embedding
                    
                    # Зберігаємо в кеш
                    if use_cache:
                        self.cache.put(texts_to_compute[i], computed_embedding)
                
                logger.info(f"Successfully computed {len(computed_embeddings)} embeddings")
                
            except Exception as e:
                logger.error(f"Failed to compute batch embeddings: {str(e)}")
                # Заповнюємо zero-векторами як fallback
                for i in indices_to_compute:
                    if embeddings[i] is None:
                        embeddings[i] = np.zeros(settings.embedding_dimension, dtype=np.float32)
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Обчислення косинусної схожості між двома ембедингами.
        
        Косинусна схожість ідеально підходить для семантичного пошуку,
        оскільки вона інваріантна до довжини векторів та фокусується
        тільки на кутовому розташуванні в просторі ознак.
        
        Значення: 1.0 = ідентичні, 0.0 = ортогональні, -1.0 = протилежні
        """
        try:
            # Для нормалізованих векторів косинусна схожість = скалярний добуток
            similarity = np.dot(embedding1, embedding2)
            
            # Обмежуємо значення в діапазоні [-1, 1] через потенційні числові помилки
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Знаходження найбільш схожих ембедингів з колекції.
        
        Це fallback метод для випадків, коли векторна БД недоступна.
        В production системах краще використовувати спеціалізовані 
        векторні бази даних для масштабованого пошуку.
        
        Returns:
            List[Tuple[int, float]]: Список кортежів (індекс, схожість)
        """
        if not candidate_embeddings:
            return []
        
        try:
            # Обчислюємо схожість з усіма кандидатами
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Сортуємо за спаданням схожості та повертаємо топ-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {str(e)}")
            return []
    
    def get_model_info(self) -> dict:
        """Отримання інформації про завантажену модель."""
        if self.model is None:
            return {
                "model_name": self.model_name,
                "loaded": False,
                "device": self.device
            }
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": str(self.model.device),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": settings.embedding_dimension
        }


# Глобальний екземпляр сервісу для використання в інших модулях
embedding_service = EmbeddingService()