import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.utils.embedding_utils import embeddings_client

from app.config.log_config import logger
from app.middleware.log_middleware import LoggingMiddleware
from app.exceptions import AppError
from app.exceptions.handlers import app_error_handler, global_exception_handler
from app.routes.ingestion_routes import router as ingestion_router
from app.routes.chat_routes import router as chat_router
from app.db.database import engine, Base

from app.config.env_config import settings
from app.repository.factory import VectorStoreFactory
from app.constants.app_constants import VECTOR_DB

@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info("asynccontextmanager")

  app.state.embeddings = embeddings_client
  db_type = settings.VECTOR_DB_TYPE

  app.state.vector_repo = VectorStoreFactory.get_repository(
    db_type = db_type,
    embeddings=app.state.embeddings,
    persist_directory=settings.VECTOR_PERSIST_DIR,
    collection_name=VECTOR_DB.COLLECTION_NAME.value
    )
  yield

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Async User API", lifespan=lifespan)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, global_exception_handler)
app.add_middleware(LoggingMiddleware)

@app.get('/')
def health():
  logger.info("🚀 Application started")
  return { "status": 'ok' }

app.include_router(ingestion_router)
app.include_router(chat_router)