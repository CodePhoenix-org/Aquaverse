from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from db.database import DB_URI, Base
from auth import routes as authroute
from auth import profile
from rag.updated_pipeline import rag_query
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

# ChromaDB configuration
CHROMA_PATH = os.getenv('CHROMA_PATH') or os.getenv('chromapath') or os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "chroma_db")
COLLECTION_NAME = os.getenv('COLLECTION_NAME') or os.getenv('collectioname') or 'argo_summaries'

os.makedirs(CHROMA_PATH, exist_ok=True)

@app.get("/")
def home():
    return {"msg": "you are awesome and connected to the database 🦖"}

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = rag_query(
            user_query=req.query.strip(),
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            db_uri=DB_URI
        )

        return {"message": result or "No relevant information found.", "data": None}

    except Exception as e:
        return {"message": f"⚠️ Error: {str(e)}", "data": None}
