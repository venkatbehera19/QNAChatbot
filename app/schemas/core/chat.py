from pydantic import BaseModel, Field, ConfigDict

class ChatRequest(BaseModel):
  """Request body for chat endpoint."""
  query: str = Field(..., description="User query")

class ChatResponse(BaseModel):
  """Response body for chat endpoint."""
  answer: str = Field(..., description="Assistant response")