from fastapi import APIRouter, UploadFile, status, File, Depends, Request, HTTPException

from app.config.env_config import settings
from app.config.log_config import logger

from app.schemas.core.ingestion import IngestionRequest
from app.services.core.ingestion_service import ingestion_service
from langchain_core.documents import Document

from app.exceptions.domain import AppError
import os


router = APIRouter(tags=["rag"])

def get_ingestion_request(file: UploadFile = File(...)) -> IngestionRequest:
  """Create an ingestion request from an uploaded file.

  Args:
    file: File uploaded via multipart form.

  Returns:
    IngestionRequest wrapping the file.
  """
  return IngestionRequest(file=file)

@router.post('/upload', status_code=status.HTTP_201_CREATED)
async def ingest_file(request: Request, file_data: IngestionRequest = Depends(get_ingestion_request)):
  """ Upload a file and index it 
  
  Args:
    request: IngestionRequest with uploaded file

  Returns:
    IngestionResponse wit message, file_path and filename

  """
  file = file_data.file
  filename = file.filename
  ingest_result = ingestion_service.save_file(file)
  service_response = ingestion_service.ingest_file(file, ingest_result)

  raw_chunks = service_response.get("index_result", [])
  vector_db = request.app.state.vector_repo

  if vector_db.file_exists(filename):
    return {
       "message": f"File '{filename}' is already indexed. Skipping ingestion.",
        "saved_path": None
    }

  try:
    chunks = []
    for item in raw_chunks:
      if isinstance(item, Document):
        original_source = item.metadata.get("source", filename)
        item.metadata["filename"] = os.path.basename(original_source)
        chunks.append(item)

    vector_db.add_documents(chunks)
    return {
      "message": "File Uploaded and Indexed Successfully",
      "saved_path":   service_response['saved_path']
    }
        
  except TypeError as e:
    logger.error(f"Mapping failed. raw_chunks type: {type(raw_chunks)}. Content: {raw_chunks[:100]}")
    raise AppError(message="Data format mismatch during indexing", status_code=500)
