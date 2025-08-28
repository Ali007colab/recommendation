import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL")
    
    # RabbitMQ
    RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
    RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
    RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
    RABBITMQ_URL = os.getenv("RABBITMQ_URL")
    
    # Service
    SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # ML
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", 384))
    MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Cache
    RECOMMENDATIONS_CACHE_TTL = int(os.getenv("RECOMMENDATIONS_CACHE_TTL", 3600))
    USER_PROFILE_CACHE_TTL = int(os.getenv("USER_PROFILE_CACHE_TTL", 1800))

config = Config()