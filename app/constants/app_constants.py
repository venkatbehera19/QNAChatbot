"""Application constants: env names, file types, model config, route paths."""

from enum import Enum

class GEMINI_CHAT_MODEL(Enum):
  """Gemini chat model configuration"""
  MODEL_NAME = "gemini-3-flash-preview"
  TEMPERATURE = 0.0

class GROQ_CHAT_MODEL(Enum):
  """Gemini chat model configuration"""
  MODEL_NAME = "openai/gpt-oss-120b"
  TEMPERATURE = 0.0

class ALLOWED_FILES(Enum):
  """Supported file extensions for ingestion."""
  PDF = ".pdf"
  DOCX = ".docx"
  ALL_FILES = (".pdf", ".docx")

class VECTOR_DB(Enum):
  """"""
  CHUNK_SIZE = 1000
  CHUNK_OVERLAP = 100
  EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

  FAISS = 'faiss'
  CHROMA = 'chroma'
  COLLECTION_NAME = "qna_indexes"

