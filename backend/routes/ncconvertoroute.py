from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from services.convertertocsv import convert_nc_to_csv
import os

router = APIRouter()

@router.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    csv_path, csv_filename = await convert_nc_to_csv(file)
    return JSONResponse(content={"convertedData": "success", "filename": csv_filename})

@router.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/csv", filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)