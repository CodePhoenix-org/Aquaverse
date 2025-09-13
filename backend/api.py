from fastapi import FastAPI
from  search_profiles import search_profiles
from sqlalchemy import create_engine
import pandas as pd
import os



app = FastAPI()
DB_URI = os.getenv("DB_URI", "postgresql://postgres:password@localhost:5432/argo_db")

@app.get("/profiles")
def getvalue(limit: int=10):
    engine = create_engine(DB_URI)
    df = pd.read_sql(f"select * from argo_profiles limit {limit}",engine)
    return df.to_dict(orient="records")


@app.get("/")
def home():
    return {"msg":"Hy Mate you are Awesome!"}



@app.get("/search")
def search(query:str,number: int =3):
    response = search_profiles(query,number=number)
    return response