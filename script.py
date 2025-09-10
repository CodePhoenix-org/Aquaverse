import os
import xarray as xr

# Path to your NetCDF file (use raw string to avoid Windows escape issues)
# file_path = r"C:\Users\Dell\Downloads\nodc_R1902344_798.nc"
file_path = r"C:\Users\Dell\Downloads\nodc_D3901888_130.nc"

# Check if file exists
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
else:
    # Open the dataset using netCDF4 engine
    ds = xr.open_dataset(file_path, engine="netcdf4")
    
    # Print summary of the dataset
    print("✅ Dataset loaded successfully!")
    print(ds)

    # Access variables, e.g. temperature
    if "TEMP" in ds.variables:
        print("\n📌 TEMP variable details:")
        print(ds["TEMP"])
    else:
        print("\n⚠️ TEMP variable not found in this dataset.")

    # Convert to dataframe (tabular format)
    df = ds.to_dataframe().reset_index()
    print("\n📊 First 5 rows of data:")
    print(df.head())
