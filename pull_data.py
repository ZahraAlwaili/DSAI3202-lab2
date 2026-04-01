from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import mltable
import os

# Your Workspace Details
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    resource_group_name="rg-60307052",
    workspace_name="Amazon-Electronics-Lab-60307052"
)

print("Fetching data from Azure... this might take a minute.")
data_asset = ml_client.data.get("amazon_review_merged_features_deploy", version="1")

# Use the mltable method to force a conversion to a local CSV/Parquet
# --- UPDATED PULL LOGIC ---
try:
    # We add the wildcard to tell Azure to look INSIDE the folder for parquet files
    folder_path = data_asset.path
    if folder_path.endswith('/'):
        pattern = f"{folder_path}**/*.parquet"
    else:
        pattern = f"{folder_path}/**/*.parquet"
        
    print(f"Searching for files in: {pattern}")
    
    tbl = mltable.from_parquet_files(paths=[{'pattern': pattern}])
    df = tbl.to_pandas_dataframe()
    
    # Save it locally
    df.to_parquet("deploy_data.parquet")
    print("\n" + "="*30)
    print("SUCCESS! 'deploy_data.parquet' is ready.")
    print(f"Total records pulled: {len(df)}")
    print("="*30)
except Exception as e:
    print(f"Failed to pull data: {e}")