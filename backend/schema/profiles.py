from pydantic import BaseModel
from datetime import datetime

class ProfileResponse(BaseModel):
    time: datetime
    latitude: float
    longitude: float
    temperature: float
    salinity: float

    class Config:
        orm_mode = True
