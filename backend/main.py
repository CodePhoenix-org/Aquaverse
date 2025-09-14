from fastapi import FastAPI
from sqlalchemy import create_engine
from db.database import DB_URI, Base, get_db
from auth import routes as authroute
from auth import profile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.rag_pipeline import rag_query






app = FastAPI(title="Aquaverse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

engine = create_engine(DB_URI)
Base.metadata.create_all(bind=engine)


app.include_router(authroute.router,prefix="/auth",tags=["Auth"])
app.include_router(profile.router, prefix="/profiles", tags=["Profiles"])


@app.get("/")
def home():
    return {"msg":"you are awesome and connected to the database🦖"}




CHROMA_PATH = 'C:/Users/Gagan/Desktop/Aquaverse/backend/mychromadb'
COLLECTION_NAME = 'argo_summaries'

# Request schema
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Handle a user query and return summarized ocean data.
    """
    try:
        result = rag_query(
            user_query=req.query,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            db_uri=DB_URI
        )
        return {"answer": result}
    except Exception as e:
        return {"error": str(e)}