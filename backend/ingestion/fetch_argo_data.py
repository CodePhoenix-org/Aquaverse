# First, make sure you have the required libraries installed:
# pip install argopy pandas
# ---------------------------------------------------------

import pandas as pd
from argopy import ArgoIndex
import json
import numpy as np

print("Fetching BGC Argo float index... This may take a moment.")

try:
    # 1. Fetch the data using argopy
    idx = ArgoIndex(index_file="bgc-b").load()
    df = idx.to_dataframe()

    # 2. Clean up the data
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    df['profiler'] = df['profiler'].astype(str)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # 3. Select only the columns needed for visualization
    output_df = df[['profiler', 'latitude', 'longitude', 'date_str']]

    # --- FIX: Replace invalid NaN values with None for valid JSON ---
    # This converts any numpy.nan or pandas.NaT into Python's None, which becomes 'null' in JSON.
    output_df = output_df.replace({np.nan: None})


    # 4. Convert the DataFrame to a list of dictionaries (JSON format)
    data_for_json = output_df.to_dict(orient='records')


    # 5. Save the data to a JSON file
    file_path = 'argodata.json'
    with open(file_path, 'w') as f:
        json.dump(data_for_json, f, indent=4)

    print(f"\n✅ Success! {len(data_for_json)} float locations saved to {file_path}")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please check your internet connection and ensure libraries are installed.")

