from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatMessageBase(BaseModel):
    sender: str      
    content: str       
    viz_data: Optional[dict] = None       
    viz_tab: Optional[str] = None

class ChatMessageCreate(BaseModel):
    # Accept both the old format and new format
    query: Optional[str] = None
    response: Optional[str] = None
    viz_type: Optional[str] = None
    
    # New format fields
    sender: Optional[str] = "user"
    content: Optional[str] = None
    viz_data: Optional[dict] = None
    viz_tab: Optional[str] = None

class ChatMessage(ChatMessageBase):
    id: int
    user_id: int       
    timestamp: datetime      

    class Config:
        from_attributes = True