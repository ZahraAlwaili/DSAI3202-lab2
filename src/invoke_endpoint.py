import pandas as pd
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.metrics import accuracy_score
import json
import requests

def main():
    # 1. Setup Connection to your UDST Lab Workspace
    credential = DefaultAzureCredential()
    
    # Using your specific lab IDs from your previous logs
    ml_client = MLClient(
        credential=credential,
        subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
        resource_group_name="rg-60307052",
        workspace_name="Amazon-Electronics-Lab-60307052"
    )

    # 2. Load the Local Data
    print("Loading local deployment data (10% split)...")
    try:
        df = pd.read_parquet("deploy_data.parquet")
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error: Could not find 'deploy_data.parquet' in this folder. {e}")
        return

    # 3. Get the Endpoint URL and Keys
    endpoint_name = "amazon-sentiment-endpoint"
    print(f"Fetching details for endpoint: {endpoint_name}")
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    keys = ml_client.online_endpoints.get_keys(endpoint_name)

    # 4. Prepare the Data for Inference
    # Separate features and the true labels
    y_true = df['sentiment'].values
    # Remove the label column so we only send features to the model
    test_data = df.drop(columns=['sentiment'])

    # Convert to JSON format for the Azure Endpoint
    sample_data = {"data": test_data.to_dict(orient='records')}
    body = str.encode(json.dumps(sample_data))

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {keys.primary_key}',
        'azureml-model-deployment': 'sentiment-model-deployment'
    }

    # 5. Call the Endpoint
    print("Sending data to the deployed model... please wait.")
    response = requests.post(endpoint.scoring_uri, data=body, headers=headers)

    if response.status_code == 200:
        predictions = json.loads(response.json())
        
        # 6. Calculate and Print Final Accuracy
        acc = accuracy_score(y_true, predictions)
        
        print("\n" + "="*40)
        print("FINAL LAB 2 RESULTS")
        print("="*40)
        print(f"Deployment Accuracy: {acc:.4f}")
        print("-" * 40)
        print(f"First 10 Predictions: {predictions[:10]}")
        print(f"First 10 Actual Labels: {list(y_true[:10])}")
        print("="*40)
        print("\nSUCCESS: Copy the Accuracy score for Section H of your report.")
    else:
        print(f"Inference failed with status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()