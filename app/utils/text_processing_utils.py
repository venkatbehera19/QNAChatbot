import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from app.config.log_config import logger
from app.constants.app_constants import VECTOR_DB
from app.config.env_config import settings
from app.utils.embedding_utils import embeddings_client

from app.constants.app_constants import ALLOWED_FILES
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

class TextProcessing:
  """load and split documents into chunk"""

  def __init__(
      self, 
      file_path, 
      chunk_size = VECTOR_DB.CHUNK_SIZE.value, 
      chunk_overlap = VECTOR_DB.CHUNK_OVERLAP.value,
      semantic_thresold_type = VECTOR_DB.SEMANTIC_CHUNKING_THRESHOLD_TYPE.value,
      semantic_thresold_amount = VECTOR_DB.SEMANTIC_CHUNKING_THRESHOLD_AMOUNT.value
      ):
    """Intialize text processor for a file
    
    Args:
      file_path: path to the document file
      chunk_size: Size of each text chunk. Defaults to VECTOR_DB setting.
      chunk_overlap: Overlap between chunks. Defaults to VECTOR_DB setting.
    """
    self.file_path = file_path
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.semantic_thresold_type = semantic_thresold_type
    self.semantic_thresold_amount = semantic_thresold_amount

  def load_documents(self):
    """Load the document from the files

    Returns:
      List of loaded Document objects.

    Raises:
      ValueError: If file extension is unsupported.
    """
    absolute_path = os.path.abspath(self.file_path)
    file_extension = os.path.splitext(self.file_path)[1].lower()

    if not os.path.exists(absolute_path):
        logger.error(f"File not found at path: {absolute_path}")
        raise FileNotFoundError(f"Docker cannot see file at {absolute_path}")
    
    try:
      if file_extension == ALLOWED_FILES.PDF.value:
        loader = PyPDFLoader(file_path=self.file_path)
        return loader.load()
      else:
        raise ValueError("Invalid file extension")
    except Exception as e:
      logger.info(f"Unable to load the doc {e}")


  def split_documents(self, docs) -> List:
    """Split the doc into chunks using either recursive charaacter splitter
      or Semantic chunking.
    Args:
      docs: List of documents loaded from the file to split.
  
    Returns:
      List of document chunks.
    """
    if settings.SEMANTIC_CHUNKING:
      text_splitter = SemanticChunker(
        embeddings= embeddings_client,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=90
      )
      logger.info("Using SemanticChunker")
    else:
      text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)
      logger.info("Using RecursiveCharacterTextSplitter")
      
    return text_splitter.split_documents(docs)

  def process(self) -> List:
    """Load and split documents in one step.
    Returns:
      List of document chunks ready for indexing.
    """
    logger.info("Inside process TextProcessing")
    docs = self.load_documents()
    split_docs = self.split_documents(docs)
    return split_docs