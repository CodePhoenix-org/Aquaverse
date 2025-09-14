from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from auth.model import Profile
from schema.profiles import ProfileResponse
from typing import List


router = APIRouter()


@router.get("/", response_model=List[ProfileResponse])
def get_profiles(db: Session = Depends(get_db)):
    profiles = db.query(Profile).limit(50).all()
    return profiles
