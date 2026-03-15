# pipeline.py
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, Environment
from azure.identity import DefaultAzureCredential

# ── Connect to Workspace ─────────────────────────────────
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    resource_group_name="rg-60307052",
    workspace_name="Amazon-Electronics-Lab-60307052"
)
print(f"✅ Connected to: {ml_client.workspace_name}")

# ── Environment ──────────────────────────────────────────
from azure.ai.ml.entities import Environment, BuildContext

env = Environment(
    name="lab5-env",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file={
        "name": "lab5-env",
        "channels": ["defaults"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "azure-storage-blob",
                    "pandas",
                    "scikit-learn",
                    "tsfresh",
                    "deap",
                    "xgboost",
                    "lightgbm",
                    "pyarrow"
                ]
            }
        ]
    }
)
# ── Components ───────────────────────────────────────────
preprocess_component = CommandComponent(
    name="preprocess",
    display_name="Preprocess Data",
    command=(
        "python components/preprocess.py "
        "--storage_account amazondatalake60307052 "
        "--account_key ${{inputs.account_key}} "
        "--container lab5 "
        "--output_path ${{outputs.output_path}}"
    ),
    inputs={"account_key": {"type": "string", "is_optional": False}},
    outputs={"output_path": {"type": "uri_folder"}},
    environment=env,
    code="./lab5_pipeline"
)

extract_component = CommandComponent(
    name="extract_features",
    display_name="Extract Features (tsfresh)",
    command=(
        "python components/extract_features.py "
        "--input_path ${{inputs.input_path}} "
        "--output_path ${{outputs.output_path}} "
        "--window_size 30"
    ),
    inputs={"input_path": {"type": "uri_folder"}},
    outputs={"output_path": {"type": "uri_folder"}},
    environment=env,
    code="./lab5_pipeline"
)

select_component = CommandComponent(
    name="select_features",
    display_name="Select Features (Filter + GA)",
    command=(
        "python components/select_features.py "
        "--input_path ${{inputs.input_path}} "
        "--output_path ${{outputs.output_path}}"
    ),
    inputs={"input_path": {"type": "uri_folder"}},
    outputs={"output_path": {"type": "uri_folder"}},
    environment=env,
    code="./lab5_pipeline"
)

train_component = CommandComponent(
    name="train_model",
    display_name="Train XGBoost Model",
    command=(
        "python components/train_model.py "
        "--input_path ${{inputs.input_path}} "
        "--output_path ${{outputs.output_path}}"
    ),
    inputs={"input_path": {"type": "uri_folder"}},
    outputs={"output_path": {"type": "uri_folder"}},
    environment=env,
    code="./lab5_pipeline"
)

# ── Pipeline ─────────────────────────────────────────────
@pipeline(
    name="lab5_predictive_maintenance",
    description="NASA C-MAPSS Predictive Maintenance Pipeline"
)
def lab5_pipeline(account_key: str):
    step1 = preprocess_component(account_key=account_key)
    step2 = extract_component(input_path=step1.outputs.output_path)
    step3 = select_component(input_path=step2.outputs.output_path)
    step4 = train_component(input_path=step3.outputs.output_path)
    return {"model_output": step4.outputs.output_path}

# ── Submit ───────────────────────────────────────────────
if __name__ == "__main__":
    ACCOUNT_KEY = "rbXHUr9voqcrJI5eLoNEFzc4ULCNRK1qI7Ogtmh2gQrGGkLnR9uDGTEN2txtGTxC43EZpQKwse9K+AStPfT+KA=="   

    pipeline_job = lab5_pipeline(account_key=ACCOUNT_KEY)
    pipeline_job.settings.default_compute = "serverless"

    job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"✅ Pipeline submitted!")
    print(f"   Job name : {job.name}")
    print(f"   Studio   : {job.studio_url}")