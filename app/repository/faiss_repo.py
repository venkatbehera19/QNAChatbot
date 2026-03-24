import os

from app.config.log_config import logger

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

class FAISSRepository:
  """ Faiss vector database configuration """
  def __init__(self, embeddings, persist_directory, collection_name):
    """ Intilize the repo
    Args:
      embeddings: embedding model for vectorization
      persist_directory: the directory we need to save the vector index file
      collection_name: name of the collection we need to store there
    """
    self.embeddings = embeddings
    self.persist_directory = persist_directory
    self.index_name = f"{collection_name}_faiss.index"
    if not os.path.exists(self.persist_directory):
      os.makedirs(self.persist_directory, exist_ok=True)
      logger.info(f"Created directory: {self.persist_directory}")
    self.vector_store = self._load_or_create()

  def _load_or_create(self):
    """"""
    if os.path.exists(os.path.join(self.persist_directory, f"{self.index_name}.faiss")):
      return FAISS.load_local(
        self.persist_directory, 
        self.embeddings, 
        index_name=self.index_name,
        allow_dangerous_deserialization=True
      )
    
    dim = len(self.embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
      self.embeddings, index, InMemoryDocstore(), {}
    )
  
  def add_documents(self, documents):
    logger.info("Adding the document statred")
    self.vector_store.add_documents(documents)
    self.vector_store.save_local(self.persist_directory, index_name=self.index_name)
    logger.info("document saved")

  def search(self, query, k=5):
    return self.vector_store.similarity_search(query, k=k)

  def file_exists(self, filename: str) -> bool:
    """Queries ChromaDB metadata to see if this filename has been processed."""
    try:
      results = self.vector_store._collection.get(
        where={"filename": filename},
        limit=1,
        include=[]  
      )
      return len(results.get("ids", [])) > 0

    except Exception as e:
      logger.info(f"Error checking Chroma for file: {e}")
      return False