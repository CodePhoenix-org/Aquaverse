from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON,func
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime



class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")


class Profile(Base):
    __tablename__ = "argo_profiles"
    id = Column(Integer, primary_key=True, index=True)
    time = Column(DateTime, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    temperature = Column(Float)
    salinity = Column(Float)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sender = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    viz_data = Column(JSON, nullable=True)
    viz_tab = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="messages")

