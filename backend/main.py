from fastapi import FastAPI
from sqlalchemy import create_engine
from db.database import DB_URI

app = FastAPI()
engine = create_engine(DB_URI)



@app.get("/")
def home():
    return {"msg":"you are awesome and connected to the database🦖","engine":str(engine.url)}
