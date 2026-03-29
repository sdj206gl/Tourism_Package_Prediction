from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

repo_id = "shashidj/tourism-package-prediction"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# ✅ Correct path handling
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

print("Resolved data path:", DATA_DIR)

if not DATA_DIR.exists():
    raise ValueError(f"❌ Data folder not found: {DATA_DIR}")

# Upload
api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=repo_id,
    repo_type=repo_type,
)

print("✅ Data uploaded successfully")
