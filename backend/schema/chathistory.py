from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatMessageBase(BaseModel):
    content: str
    sender: str
    viz_data: Optional[dict] = None
    viz_tab: Optional[str] = None

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessage(ChatMessageBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True   
