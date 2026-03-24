from fastapi import UploadFile
from app.config.log_config import logger
from app.exceptions import InternalServerError, ValidationError
from typing import Optional

from app.utils.file_utils import FileProcessor
from app.utils.text_processing_utils import TextProcessing

class IngestionService:
  """This service is used for file storage and indexing"""

  def save_file(self, file:UploadFile) -> str:
    """Save the file to the upload directory
    
    Args:
      file: the uploaded file to save

    Returns: 
      path where the file was save

    Raises:
      ValidationError: If file save fails.
      InternalError: If unexpected error occurs.
    """
    try:
      file_processor = FileProcessor(file=file)
      file_processor.get_file_name()
      file_processor.get_file_extension()
      file_path = file_processor.get_file_path()
      saved_path = file_processor.save_file(file_path)
      return saved_path

    except (OSError, IOError) as e:
      logger('ERRROR while saving the file:  %s', e)
      raise ValidationError('File Upload Fail') from e
    except Exception as e:
      logger.exception("Unexpected error saving file: %s", e)
      raise InternalServerError('file upload fail') from e
    
  def ingest_file(self, file: UploadFile, saved_path: Optional[str] = None) -> str:
    """Save the file (if needed) and index it into the vector database."""
    
    try:
      if saved_path is None:
        saved_path = self.save_file(file)
      logger.info('Working here ingest_file')
      index_result = self.index_file(saved_path)
      return {"saved_path": saved_path, "index_result": index_result}
    except (ValidationError, InternalServerError):
      raise
    except Exception as e:
      logger.exception("Ingestion/indexing failed: %s", e)
      raise InternalServerError("File ingestion failed") from e
    
  def index_file(self, file_path: str):
    """Index an already-saved file into the vector database
    
    Args:
      file_path: Path to saved file

    Returns:
      Index tool result
    """
    logger.info('Working here index_file')
    processor = TextProcessing(file_path)
    results = processor.process()
    
    return results

ingestion_service = IngestionService()