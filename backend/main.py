from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from db.database import DB_URI, Base
from auth import routes as authroute
from auth import profile
from routes import chat,ncconvertoroute
from routes import threeD_visualisations
from rag.dataVisualisation import rag_query
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()


app = FastAPI(title="Aquaverse Project APIs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DB_URI)
Base.metadata.create_all(bind=engine)

app.include_router(authroute.router, prefix="/auth", tags=["Auth"])
app.include_router(profile.router, prefix="/profiles", tags=["Profiles"])
app.include_router(chat.router, tags=["Chat"]) 
app.include_router(ncconvertoroute.router, tags=["NC Convertor"])
app.include_router(threeD_visualisations.router)


# ChromaDB configuration
# CHROMA_PATH = os.getenv('chromapath') 
# COLLECTION_NAME = os.getenv('collectioname') 
CHROMA_PATH = os.getenv("chromapath")
COLLECTION_NAME = os.getenv("collectionname")

os.makedirs(CHROMA_PATH, exist_ok=True)

@app.get("/")
def home():
    return {"msg": "you are awesome and connected to the database 🦖"}

class ChatRequest(BaseModel):
    query: str

@app.post("/chat/query")
async def chat(req: ChatRequest):
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        

        # This should return a tuple (message, data)
        message, data = rag_query(
            user_query=req.query.strip(),
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            db_uri=DB_URI
        )

        return {"message": message, "data": data}

    except Exception as e:
        return {"message": f"⚠️ Error: {str(e)}", "data": None}
