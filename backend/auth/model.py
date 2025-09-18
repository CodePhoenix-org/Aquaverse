from sqlalchemy import Column,Integer,String,Float,DateTime
from db.database import Base
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime
from db.database import Base
from datetime import datetime



class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Profile(Base):
    __tablename__ = "argo_profiles"
    id = Column(Integer, primary_key=True, index=True)
    time = Column(DateTime, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    temperature = Column(Float)
    salinity = Column(Float)


class ChatRequest(BaseModel):
    message:str