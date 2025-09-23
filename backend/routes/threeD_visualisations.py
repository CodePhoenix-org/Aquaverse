from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import json
from db.database import DB_URI

router = APIRouter(prefix="/api/3d", tags=["3D Visualizations"])

@router.get("/parameter-plot")
async def get_3d_parameter_plot(region: str = "indian", limit: int = 1000):
    """Returns data for the Temperature-Salinity-Oxygen 3D plot"""
    try:
        # Build SQL query based on region
        if region.lower() == "indian":
            where_conditions = "latitude BETWEEN -40 AND 30 AND longitude BETWEEN 20 AND 120"
        else:
            where_conditions = "1=1"  # All data
        
        sql = f"""
        SELECT temperature, salinity, oxygen, 
               CASE 
                   WHEN temperature > 25 OR salinity < 34 OR salinity > 36 OR oxygen < 180 THEN 'Anomaly'
                   ELSE 'Normal'
               END as anomaly_status
        FROM argo_profiles
        WHERE {where_conditions}
          AND temperature IS NOT NULL 
          AND salinity IS NOT NULL 
          AND oxygen IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """
        
        engine = create_engine(DB_URI)
        df = pd.read_sql(text(sql), engine)
        
        # Split into normal and anomaly data
        df_normal = df[df['anomaly_status'] == 'Normal']
        df_anomaly = df[df['anomaly_status'] == 'Anomaly']
        
        return {
            "type": "3d_parameter_plot",
            "normalData": df_normal[['temperature', 'salinity', 'oxygen']].rename(columns={
                'temperature': 'Temperature',
                'salinity': 'Salinity', 
                'oxygen': 'Oxygen'
            }).to_dict('records'),
            "anomalyData": df_anomaly[['temperature', 'salinity', 'oxygen']].rename(columns={
                'temperature': 'Temperature',
                'salinity': 'Salinity',
                'oxygen': 'Oxygen'
            }).to_dict('records'),
            "metadata": {
                "title": "3D Parameter Comparison",
                "region": region,
                "total_points": len(df),
                "normal_count": len(df_normal),
                "anomaly_count": len(df_anomaly)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D parameter data: {str(e)}")

@router.get("/ocean-plot")
async def get_3d_ocean_plot(region: str = "indian", limit: int = 500):
    """Returns data for the Ocean 3D plot with stems"""
    try:
        # Build SQL query based on region
        if region.lower() == "indian":
            where_conditions = "latitude BETWEEN -40 AND 30 AND longitude BETWEEN 20 AND 120"
        else:
            where_conditions = "1=1"  # All data
        
        sql = f"""
        SELECT longitude, latitude, depth, temperature,
               CASE 
                   WHEN temperature > 25 OR depth > 1000 THEN 'Anomaly'
                   ELSE 'Normal'
               END as anomaly_status
        FROM argo_profiles
        WHERE {where_conditions}
          AND longitude IS NOT NULL 
          AND latitude IS NOT NULL 
          AND depth IS NOT NULL
          AND temperature IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """
        
        engine = create_engine(DB_URI)
        df = pd.read_sql(text(sql), engine)
        
        return {
            "type": "3d_ocean_plot",
            "oceanData": df.rename(columns={
                'longitude': 'Longitude',
                'latitude': 'Latitude', 
                'depth': 'Depth',
                'temperature': 'Temperature',
                'anomaly_status': 'Anomaly_Status'
            }).to_dict('records'),
            "metadata": {
                "title": "3D Ocean Data Visualization",
                "region": region,
                "total_points": len(df)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D ocean data: {str(e)}")