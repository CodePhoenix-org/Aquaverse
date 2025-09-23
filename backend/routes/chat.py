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
        .order_by(ChatMessageModel.timestamp.asc())\
        .all()

@router.post("/history", response_model=ChatMessage)
def post_chat_message(
    chat: ChatMessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_msg = ChatMessageModel(
        user_id=current_user.id,
        sender=chat.sender,
        content=chat.content,
        viz_data=chat.viz_data,
        viz_tab=chat.viz_tab
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    return new_msg
