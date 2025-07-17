import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
BASE_DIR = Path(__file__).parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
VOLCARK_API_KEY = os.getenv("VOLCARK_API_KEY")
MODEL_NAME_MAPPING = {"deepseek-r1": "ep-20250327161916-586pz"}
BOCHA_TOKEN = os.getenv("BOCHA_TOKEN")
RAG_HOST = os.getenv("RAG_HOST")
RAG_PORT = os.getenv("RAG_PORT")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")

# MongoDB configuration
MONGODB_URI = os.getenv("DBURL")
MONGODB_DB_NAME = "pxplore"

# 打印参数配置
def print_config():
    print("Configuration:")
    print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
    print(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MONGODB_URI: {MONGODB_URI}")
    print(f"MONGODB_DB_NAME: {MONGODB_DB_NAME}")

print_config()

