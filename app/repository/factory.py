from .faiss_repo import FAISSRepository
from .chroma_repo import ChromaRepository
from .qdrant_repo import QdrantRepository

from app.constants.app_constants import VECTOR_DB
from app.config.env_config import settings
from app.utils.embedding_utils import embeddings_client

class VectorStoreFactory:
  """Dynamically selects and initializes the vector DB repository based on settings."""

  def __init__(self):
    """Intializing the varaibles that will be used inside the class"""
    self._repositories = {
      VECTOR_DB.FAISS.value: FAISSRepository,
      VECTOR_DB.CHROMA.value: ChromaRepository,
      VECTOR_DB.QDRANT.value: QdrantRepository
    }
    self.db_type = settings.VECTOR_DB_TYPE.lower()
    self.embedding_client = embeddings_client
    self.persist_directory = settings.VECTOR_PERSIST_DIR
    self.collection_name = VECTOR_DB.COLLECTION_NAME.value

  def get_repository(self):
    """return the vector store according to the configuration."""
    repo_class = self._repositories.get(self.db_type)

    if not repo_class:
      raise ValueError(f"Unsupported Vector DB type: {self.db_type}")
    
    if self.db_type == VECTOR_DB.QDRANT.value:
      return repo_class(
        embeddings =self.embedding_client,
        collection_name=self.collection_name
      )
    
    return repo_class(
      embeddings=embeddings_client,
      persist_directory=self.persist_directory,
      collection_name=self.collection_name
    )

  # @staticmethod
  # def get_repository(db_type, embeddings, persist_directory, collection_name):
  #   db_type = db_type.lower()
  #   if db_type == VECTOR_DB.FAISS.value:
  #     return FAISSRepository(
  #       embeddings= embeddings,
  #       persist_directory=persist_directory,
  #       collection_name=collection_name
  #     )
  #   elif db_type == VECTOR_DB.CHROMA.value:
  #     return ChromaRepository(
  #       embeddings=embeddings,
  #       persist_directory=persist_directory,
  #       collection_name=collection_name
  #     )
  #   elif db_type == VECTOR_DB.QDRANT.value:
  #     return QdrantRepository(
  #       collection_name=collection_name,
  #       embeddings=embeddings,
  #     )
  #   else:
  #     raise ValueError(f"Unsupported Vector DB type: {db_type}")

vector_database = VectorStoreFactory().get_repository()