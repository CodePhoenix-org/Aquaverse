from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatMessageBase(BaseModel):
    sender: str 
    content: str  
    viz_data: Optional[dict] = None  
    viz_tab: Optional[str] = None  

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessage(ChatMessageBase):
    id: int
    user_id: int  
    timestamp: datetime

    class Config:
        from_attributes = True  