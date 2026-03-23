from fastapi import APIRouter, status, Request
from langchain_core.output_parsers import StrOutputParser

from app.config.env_config import settings
from app.config.log_config import logger

# from app.schemas.core.chat import ChatRequest, ChatResponse
from app.llm.gemini_chat_client import default_chat_client

router = APIRouter(tags=["chat"])

@router.get('/chat', status_code=status.HTTP_200_OK)
async def chat(request: Request, query: str):
  """ Chat Methods for llm response

  Args:
    chat_request: 
      query: user message to llm

  Response:
    answer: llm answer with similarity search
  """
  vector_db = request.app.state.vector_repo
  if vector_db.vector_store is None:
    logger.warning("Search attempted on an empty vector store.")
    return {
      "query": query,
      "results": [],
      "message": "No documents have been indexed yet."
    }
  vector_db  = request.app.state.vector_repo
  docs = vector_db.search(query)
  context_text = "\n\n".join([doc.page_content for doc in docs])
  chain = default_chat_client | StrOutputParser()
  messages = [
      (
         "system",
          f"""You are a helpful assistant. Use the following pieces of retrieved 
          context to answer the question. If you don't know the answer based 
          on the context, say that you don't know.
            
          CONTEXT:
          {context_text}"""
      ),
      ("human", query),
  ]
  response_text = chain.invoke(messages)
  return {
    "answer": response_text,
    "sources": [doc.metadata for doc in docs]
  }

