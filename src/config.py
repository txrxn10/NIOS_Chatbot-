import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configurations
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Amazon First-Party Models
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "apac.amazon.nova-lite-v1:0"

# PostgreSQL (Docker Port 5433)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "nios_chatbot"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5433)),
}