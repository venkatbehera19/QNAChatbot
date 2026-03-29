from fastapi import APIRouter, UploadFile, status, File, Depends, Request, HTTPException
from langchain_core.documents import Document

from app.config.env_config import settings
from app.config.log_config import logger
from app.exceptions.domain import AppError
from app.constants.app_constants import VECTOR_DB
from app.repository.factory import vector_database

from app.schemas.core.ingestion import IngestionRequest
from app.services.core.ingestion_service import ingestion_service

import os
import time

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
async def ingest_file(file_data: IngestionRequest = Depends(get_ingestion_request)):
  """ Upload a file and index it 
  
  Args:
    request: IngestionRequest with uploaded file

  Returns:
    IngestionResponse wit message, file_path and filename

  """
  file = file_data.file
  filename = file.filename
  ingest_result = ingestion_service.save_file(file)
  start_time = time.perf_counter()
  service_response = ingestion_service.ingest_file(file, ingest_result)
  end_time = time.perf_counter()

  chunking_duration_ms = round((end_time - start_time) * 1000, 2)
  raw_chunks = service_response.get("index_result", [])

  if vector_database.file_exists(filename):
    return {
      "message": f"File '{filename}' uploaded and indexed successfully in {chunking_duration_ms}ms.",
      "saved_path": None,
      "chunking_time_ms": 0
    }

  try:
    chunks = []
    for item in raw_chunks:
      if isinstance(item, Document):
        original_source = item.metadata.get("source", filename)
        item.metadata["filename"] = os.path.basename(original_source)
        chunks.append(item)

    vector_database.add_documents(chunks)
    return {
      "message": "File Uploaded and Indexed Successfully",
      "saved_path":   service_response['saved_path'],
      "chunking_time_ms": chunking_duration_ms
    }
        
  except TypeError as e:
    logger.error(f"Mapping failed. raw_chunks type: {type(raw_chunks)}. Content: {raw_chunks[:100]}")
    raise AppError(message="Data format mismatch during indexing", status_code=500)
