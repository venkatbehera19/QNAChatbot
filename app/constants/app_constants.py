"""Application constants: env names, file types, model config, route paths."""

from enum import Enum
class Tax(Enum):
  """constants tax value"""
  NEW_REGIME_STANDARD_DEDUCTION = 70000.00
  OLD_REGIME_STANDARD_DEDUCTION = 50000.00


class GEMINI_CHAT_MODEL(Enum):
  """Gemini chat model configuration"""
  MODEL_NAME = "gemini-3-flash-preview"
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
  EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

  FAISS = 'faiss'
  CHROMA = 'chroma'
  COLLECTION_NAME = "qna_indexes"


