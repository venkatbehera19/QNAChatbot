from .faiss_repo import FAISSRepository
from .chroma_repo import ChromaRepository

from app.constants.app_constants import VECTOR_DB

class VectorStoreFactory:
  """"""
  @staticmethod
  def get_repository(db_type, embeddings, persist_directory, collection_name):
    db_type = db_type.lower()
    if db_type == VECTOR_DB.FAISS.value:
      return FAISSRepository(
        embeddings= embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
      )
    elif db_type == VECTOR_DB.CHROMA.value:
      return ChromaRepository(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
      )
    else:
      raise ValueError(f"Unsupported Vector DB type: {db_type}")


