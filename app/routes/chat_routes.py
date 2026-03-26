from fastapi import APIRouter, status, Request, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config.env_config import settings
from app.config.log_config import logger
from app.prompt.retrival_system_prompt import RAG_SYSTEM_PROMPT_TEXT
from app.llm.groq_chat_client import default_chat_client
from app.utils.redis_utils import redis_history
from app.config.redis_config import redis_config
# import redis

router = APIRouter(tags=["chat"])

store = {}
def get_session_history(session_id: str):
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]

@router.get('/chat', status_code=status.HTTP_200_OK)
async def chat(request: Request, query: str, session_id: str):
  """ Chat Methods for llm response

  Args:
    chat_request: 
      query: user message to llm

  Response:
    answer: llm answer with similarity search
  """

  prompt = ChatPromptTemplate.from_messages(
    [
      (
        "system", 
        RAG_SYSTEM_PROMPT_TEXT
      ),
      MessagesPlaceholder(variable_name="history"),
      ("human", "{query}")
    ]
  )
  
  vector_db = request.app.state.vector_repo
  if vector_db.vector_store is None:
    logger.warning("Search attempted on an empty vector store.")
    return {
      "query": query,
      "results": [],
      "message": "No documents have been indexed yet."
    }
    
  docs = vector_db.search(query)
  context_text = "\n\n".join([doc.page_content for doc in docs])
  # chain = default_chat_client | StrOutputParser()
  basic_chain = prompt | default_chat_client | StrOutputParser()

  if settings.USE_REDIS:
    with_message_history_runnable = RunnableWithMessageHistory(
      basic_chain,
      get_session_history=redis_history.get_redis_history,
      input_messages_key="query",
      history_messages_key="history",
    )
  else:
    with_message_history_runnable = RunnableWithMessageHistory(
      basic_chain,
      get_session_history,
      input_messages_key="query",
      history_messages_key="history",
    )
  
  config = {"configurable": { "session_id": session_id }}

  try:
    response_text = await with_message_history_runnable.ainvoke(
      {"query": query, "context": context_text}, 
      config=config
    )
    return {
      "answer": response_text,
      "sources": [doc.metadata for doc in docs]
    }
  except Exception as e:
    logger.error(f"Chain error: {e}")
    raise
  

@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
  """Fetch the complete chat history for a specific session from Redis.
  
  Args:
    session_id: the key that fetches from redis in memory
  """
  try:
    history = redis_history.get_redis_history(session_id)
    messages = history.messages

    # url =  redis_config.get_redis_url()
    # client = redis.from_url(url, decode_responses=True)
    # keys = client.keys("rag_chat:*")
    # logger.info(f"KEYS: {keys}")

    formatted_history = []
    for msg in messages:
      formatted_history.append({
        "role": "user" if msg.type == "human" else "assistant",
        "content": msg.content
      })

    return {
      "session_id": session_id,
      "history": formatted_history,
      "count": len(formatted_history)
    }

  except Exception as e:
    logger.error(f"Error fetching history for session {session_id}: {e}")
    raise HTTPException(status_code=500, detail="Could not retrieve chat history.")