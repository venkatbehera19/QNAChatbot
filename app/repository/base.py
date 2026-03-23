from abc import ABC, abstractmethod

class VectorRepository(ABC):
  """Base class for every vector db repo"""
  @abstractmethod
  def add_documents(self, documents):
    pass

  @abstractmethod
  def search(self, query, k=5):
    pass