from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

# Your Workspace Details
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    resource_group_name="rg-60307052",
    workspace_name="Amazon-Electronics-Lab-60307052"
)

print("Attempting to download data asset...")
data_asset = ml_client.data.get("amazon_review_merged_features_deploy", version="1")

# Use the MLClient's built-in download feature
try:
    # This will create a folder called 'downloaded_data' and put the file inside
    ml_client.data.download(name="amazon_review_merged_features_deploy", version="1", download_path="./downloaded_data")
    print("Download successful! Checking for the file...")
    
    # Let's find the file and move it to the main folder
    for root, dirs, files in os.walk("./downloaded_data"):
        for file in files:
            if file.endswith(".parquet"):
                old_path = os.path.join(root, file)
                new_path = "deploy_data.parquet"
                os.replace(old_path, new_path)
                print(f"File moved to: {new_path}")
                break
except Exception as e:
    print(f"Download failed: {e}")