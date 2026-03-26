import os
from app.config.log_config import logger
from app.config.env_config import settings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from app.utils.embedding_utils import embeddings_client

class QdrantRepository:
  """Qdrant vector database configuration"""
  def __init__(self, embeddings, collection_name, host = settings.QDRANT_HOST, port=settings.QDRANT_PORT, path = None):
    """Initialize the repo
    Args:
      embeddings: embedding model for vectorization
      collection_name: name of the collection
      host: Qdrant service name in docker-compose
      port: Qdrant port (default 6333)
      path: If using local storage instead of server (optional)
    """
    self.embeddings = embeddings
    self.collection_name = collection_name

    if path:
      self.client = QdrantClient(path=path)
      logger.info(f"Qdrant initialized locally at {path}")
    else:
      self.client = QdrantClient(host=host, port=port)
      logger.info(f"Qdrant connecting to {host}:{port}")

    if not self.client.collection_exists(self.collection_name):
      self.client.create_collection(
        collection_name= self.collection_name,
        vectors_config= VectorParams(
          size= len(embeddings_client.embed_query("Hello world")),
          distance= Distance.COSINE
        )
      )
      logger.info(f"Created new Qdrant collection: {self.collection_name}")

    self.vector_store = QdrantVectorStore(
      client=self.client,
      collection_name=self.collection_name,
      embedding=self.embeddings,
    )

  def add_documents(self, documents):
    logger.info(f"Adding {len(documents)} documents to Qdrant...")
    self.vector_store.add_documents(documents)
    logger.info("Successfully indexed documents.")

  def search(self, query, k=5):
    return self.vector_store.similarity_search(query, k=k)

  def file_exists(self, filename: str) -> bool:
    """Efficiently queries Qdrant metadata using the Scroll API."""
    try:
      results, _ = self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=models.Filter(
          must=[
            models.FieldCondition(
              key="metadata.filename",
              match=models.MatchValue(value=filename),
            )
          ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False
      )
    except Exception as e:
      logger.error(f"Error checking Qdrant for file: {e}")
      return False