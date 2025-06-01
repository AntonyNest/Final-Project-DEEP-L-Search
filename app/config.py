# Configuration settings

## Центральна конфігурація системи пошуку документів.

# Цей модуль реалізує pattern "Configuration as Code" 
# та забезпечує централізоване управління всіма налаштуваннями системи 
# через змінні середовища.


from pydantic import BaseSettings, Field
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Головний клас налаштувань системи.
    
    Використовуємо Pydantic Settings для автоматичної валідації 
    та type safety конфігурації. Це критично важливо для ML систем,
    де неправильна конфігурація може призвести до тихих помилок.
    """
    
    # === Database Configuration ===
    # Qdrant connection settings - separating host and port allows for flexible deployment
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant REST API port")
    
    # Collection name - using semantic naming for better maintainability
    qdrant_collection_name: str = Field(
        default="documents", 
        description="Name of the Qdrant collection for document vectors"
    )
    
    # === Document Processing Configuration ===
    # Path to documents - this is where your .doc/.docx files are located
    documents_path: str = Field(
        default=r"C:\Users\Nesterenko Anton\Desktop\МІА ОІЗ",
        description="Local path to documents directory"
    )
    
    # Text chunking parameters - critical for retrieval quality
    # Smaller chunks = more precise matching but potential context loss
    # Larger chunks = better context but less precise matching
    max_chunk_size: int = Field(
        default=1000, 
        description="Maximum size of text chunks for embedding",
        ge=100,  # Minimum 100 characters per chunk
        le=8000  # Maximum to stay within model limits
    )
    
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks to preserve context",
        ge=0
    )
    
    # === ML Model Configuration ===
    # Using multilingual model because your documents are likely in Ukrainian
    # This model supports 50+ languages including Ukrainian and Russian
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="HuggingFace model name for text embeddings"
    )
    
    # Model parameters for fine-tuning performance vs quality trade-off
    embedding_dimension: int = Field(
        default=384,  # Dimension of the chosen model
        description="Dimension of embedding vectors"
    )
    
    # Device selection - automatically detect GPU if available
    device: str = Field(
        default="auto",
        description="Device for model inference: 'auto', 'cpu', 'cuda'"
    )
    
    # === Search Configuration ===
    # Default parameters for vector search
    default_limit: int = Field(
        default=10,
        description="Default number of search results to return",
        ge=1,
        le=100
    )
    
    # Score threshold for semantic similarity
    # 0.0 = no filtering, 1.0 = exact match only
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for search results",
        ge=0.0,
        le=1.0
    )
    
    # === API Configuration ===
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    
    # CORS settings for web frontend integration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # === Logging Configuration ===
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    log_file: Optional[str] = Field(
        default="logs/document_search.log",
        description="Path to log file"
    )
    
    # === Application Metadata ===
    app_name: str = Field(default="Document Search Service", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    
    class Config:
        """Pydantic configuration for environment variable loading."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def documents_path_obj(self) -> Path:
        """
        Convert documents path string to Path object with validation.
        
        This property ensures that the path exists and is accessible,
        which is critical for document processing.
        """
        path = Path(self.documents_path)
        if not path.exists():
            raise ValueError(f"Documents path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Documents path is not a directory: {path}")
        return path
    
    @property
    def qdrant_url(self) -> str:
        """Generate complete Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    def validate_ml_config(self) -> bool:
        """
        Validate ML-specific configuration parameters.
        
        This method checks if the embedding model exists and if
        the chunk size is compatible with the model's context window.
        """
        # Validate chunk size relative to model context
        if self.max_chunk_size > 6000:  # Conservative limit for most models
            print("Warning: Large chunk size may exceed model context window")
        
        # Validate overlap settings
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        return True


# Global settings instance - singleton pattern for configuration
settings = Settings()

# Validate configuration on module import
settings.validate_ml_config()