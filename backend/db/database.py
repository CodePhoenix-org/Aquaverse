from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine


load_dotenv()

DBUSER = os.getenv("dbuser")
DBPASS = quote_plus(os.getenv("dbpass"))
DBHOST = os.getenv("dbhost")
DBPORT = os.getenv("dbport")
DBNAME = os.getenv("dbname")

DB_URI = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}"

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()