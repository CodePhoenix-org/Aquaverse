from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from schema.chathistory import ChatMessageCreate, ChatMessage
from auth.model import ChatMessage as ChatMessageModel
from auth.utils import get_current_user
from auth.model import User
from typing import List

router = APIRouter(prefix="/chat")

@router.get("/history", response_model=List[ChatMessage])
def get_chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return db.query(ChatMessageModel)\
        .filter(ChatMessageModel.user_id == current_user.id)\
        .order_by(ChatMessageModel.timestamp.desc())\
        .all()

@router.post("/history", response_model=ChatMessage)
def post_chat_message(
    chat: ChatMessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Handle both old and new formats
    if chat.query and chat.response:
        # Old format from frontend
        content = f"Q: {chat.query}\nA: {chat.response}"
        sender = "user"  # Default sender
        viz_data = None
        viz_tab = chat.viz_type
    else:
        # New format
        content = chat.content or ""
        sender = chat.sender or "user"
        viz_data = chat.viz_data
        viz_tab = chat.viz_tab
    
    new_msg = ChatMessageModel(
        user_id=current_user.id,
        sender=sender,
        content=content,
        viz_data=viz_data,
        viz_tab=viz_tab
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    return new_msg