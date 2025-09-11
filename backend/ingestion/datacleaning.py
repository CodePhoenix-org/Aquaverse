import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


ds = xr.open_dataset('../data/raw/7902287_prof.nc')
print("tuples in dataset:", list(ds.variables))



#selected particular tuples (column values)
lat = ds["LATITUDE"].values
lon = ds["LONGITUDE"].values
juld = ds["JULD"].values
pres = ds["PRES"].values
temp = ds["TEMP"].values
psal = ds["PSAL"].values







# refernce date for current date (julid format)
ref_date = datetime(1950, 1, 1)

if np.issubdtype(juld.dtype, np.datetime64):
    time = pd.to_datetime(juld)
else:
    # Otherwise treat as "days since 1950-01-01"
    juld = juld.astype(float)  # force float
    juld = np.where((np.isfinite(juld)) & (juld < 100000), juld, np.nan)  # filter invalid
    time = [
        ref_date + timedelta(days=float(t)) if not np.isnan(t) else pd.NaT
        for t in juld
    ]

# no.of profiles / no.of levels
n_prof, n_levels = pres.shape
lat_expanded = np.repeat(lat, n_levels)
lon_expanded = np.repeat(lon, n_levels)
time_expanded = np.repeat(time, n_levels)






# creating dataframe  ( for a new fresh csv for this selected tuples )
df = pd.DataFrame({
    "time": time_expanded,
    "latitude": lat_expanded,
    "longitude": lon_expanded,
    "pressure": pres.flatten(),
    "temperature": temp.flatten(),
    "salinity": psal.flatten(),

})
 

 # all null values will be dropped 
df = df.dropna()

# Save as CSV
df.to_csv("content.csv", index=False)