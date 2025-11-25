"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "KG Agent Crawler"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API
    API_V1_PREFIX: str = "/api/v1"
    API_KEY_NAME: str = "X-API-Key"
    API_KEY_SECRET: str = "changeme"
    CORS_ORIGINS: List[str] = ["*"]

    # Crawler
    CRAWLER_MAX_CONCURRENT: int = 5
    CRAWLER_TIMEOUT: int = 30
    CRAWLER_USER_AGENT: Optional[str] = None
    CRAWLER_HEADLESS: bool = True
    CRAWLER_CACHE_DIR: str = "./cache/crawler"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Storage
    STORAGE_DIR: str = "./storage"
    SCREENSHOT_DIR: str = "./storage/screenshots"
    PDF_DIR: str = "./storage/pdfs"
    DOCUMENTS_DB_PATH: str = "./storage/documents.db"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "document_chunks"

    # Models
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_HOME: str = "./models/embeddings"
    EMBEDDING_DIM: int = 384  # Dimension for all-MiniLM-L6-v2

    # Graph Database (Graphiti)
    # Supports: "neo4j" or "falkordb"
    GRAPH_DRIVER: str = "falkordb"
    
    # Neo4j settings (if GRAPH_DRIVER=neo4j)
    NEO4J_URI: str = "bolt://localhost:7687"  # Use bolt:// for single instance
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # FalkorDB settings (if GRAPH_DRIVER=falkordb)
    FALKORDB_HOST: str = "localhost"
    FALKORDB_PORT: int = 6380
    FALKORDB_PASSWORD: str = "password"

    # LM Studio / Local LLM
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_API_KEY: str = "lm-studio"  # LM Studio doesn't require a real key
    LLM_MODEL_NAME: str = "local-model"  # Model name loaded in LM Studio
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # Pydantic AI Agent
    AGENT_SYSTEM_PROMPT: str = """You are a Knowledge Graph Agent that helps users explore and query a knowledge base.
You have access to tools that can:
1. Search the vector database (ChromaDB) for semantically similar content
2. Search the knowledge graph (Neo4j) for structured information
3. Start web crawls to add new content to the knowledge base
4. Get statistics about the current state of the databases

Always provide helpful, accurate responses based on the information retrieved from these tools.
When answering questions, cite the sources when available."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()
