"""Configuration management for AREIP."""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    zillow_api_key: Optional[str] = Field(None, env="ZILLOW_API_KEY")
    rentcast_api_key: Optional[str] = Field(None, env="RENTCAST_API_KEY")
    walk_score_api_key: Optional[str] = Field(None, env="WALK_SCORE_API_KEY")
    fred_api_key: Optional[str] = Field(None, env="FRED_API_KEY")
    
    # Database Connections
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("areip", env="POSTGRES_DB")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: str = Field("postgres", env="POSTGRES_PASSWORD")
    
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Google Cloud Platform
    gcp_project_id: Optional[str] = Field(None, env="GCP_PROJECT_ID")
    gcp_region: str = Field("us-central1", env="GCP_REGION")
    google_application_credentials: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Observability
    langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # Application Settings
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    # Geographic Settings
    default_city: str = Field("San Francisco", env="DEFAULT_CITY")
    default_state: str = Field("CA", env="DEFAULT_STATE")
    default_country: str = Field("US", env="DEFAULT_COUNTRY")
    
    # Data Sources Configuration
    data_update_interval_hours: int = Field(24, env="DATA_UPDATE_INTERVAL_HOURS")
    max_properties_per_batch: int = Field(1000, env="MAX_PROPERTIES_PER_BATCH")
    enable_synthetic_data: bool = Field(True, env="ENABLE_SYNTHETIC_DATA")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_postgres_url(self) -> str:
        """Get async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


# Global settings instance
settings = Settings()