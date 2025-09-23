import os
import uuid
import pandas as pd
import xarray as xr
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def convert_nc_to_csv(file: UploadFile):
    # Validate extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != ".nc":
        raise ValueError("Only .nc files are allowed")

    # Save uploaded file temporarily
    tmp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.nc")
    with open(tmp_file_path, "wb") as f:
        f.write(await file.read())

    # Load NetCDF and convert to DataFrame
    ds = xr.open_dataset(tmp_file_path)
    df = ds.to_dataframe().reset_index()

    # Save as CSV
    csv_filename = os.path.splitext(file.filename)[0] + ".csv"
    csv_path = os.path.join(UPLOAD_DIR, csv_filename)
    df.to_csv(csv_path, index=False)

    return csv_path, csv_filename
