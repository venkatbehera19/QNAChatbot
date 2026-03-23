import os
from app.config.log_config import logger
from langchain_chroma import Chroma

class ChromaRepository:
  """ Chroma vector database configuration """
  def __init__(self, embeddings, persist_directory, collection_name):
    """ Intilize the repo
    Args:
      embeddings: embedding model for vectorization
      persist_directory: the directory we need to save the vector index file
      collection_name: name of the collection we need to store there
    """
    self.embeddings = embeddings
    self.persist_directory = persist_directory
    self.collection_name = collection_name
        
    os.makedirs(self.persist_directory, exist_ok=True)
        
    # Chroma handles loading/creating automatically
    self.vector_store = Chroma(
        collection_name=self.collection_name,
        embedding_function=self.embeddings,
        persist_directory=self.persist_directory
    )
    logger.info(f"ChromaDB initialized in {self.persist_directory}")

  def add_documents(self, documents):
    logger.info("Adding documents to ChromaDB...")
    self.vector_store.add_documents(documents)
    # Note: In newer langchain-chroma versions, persist() is called automatically
    logger.info(f"Successfully indexed {len(documents)} documents.")

  def search(self, query, k=5):
    return self.vector_store.similarity_search(query, k=k)