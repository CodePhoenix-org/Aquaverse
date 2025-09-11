import os
import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import chromadb
from sentence_transformers import SentenceTransformer

#main py file for database and other 

RAW_DATA_PATH = '../data/raw/7902287_prof.nc'
PROCESSED_PARQUET_PATH = '../../data/processed/argo_profiles.parquet'
DB_URI = 'postgresql://postgres:Aashi@1234@localhost:5432/argo_db'
CHROMA_PATH = '../db/chroma_db'
CHROMA_COLLECTION_NAME = 'argo_summaries'

    