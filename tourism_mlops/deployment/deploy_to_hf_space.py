
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

def deploy_to_huggingface_space():
    """
    Deploy Tourism Package Prediction application to HuggingFace Spaces
    
    This script:
    1. Creates a HuggingFace Space repository
    2. Uploads all deployment files (Dockerfile, app.py, requirements.txt)
    3. Configures the space for automatic deployment
    """
    
    # Configuration
    SPACE_NAME = "Tourism-Package-Prediction"
    REPO_ID = f"shashidj/{SPACE_NAME}"
    
    print("🚀 Starting deployment to HuggingFace Spaces...")
    print(f"📦 Target Space: {REPO_ID}")
    
    # Initialize HuggingFace API client
    try:
        api = HfApi(token=os.getenv("HF_TOKEN"))
        print("✅ HuggingFace API client initialized")
    except Exception as e:
        print(f"❌ Error initializing HuggingFace API: {e}")
        print("💡 Please ensure HF_TOKEN environment variable is set")
        return False
    
    # Check if space already exists
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="space")
        print(f"📍 Space '{REPO_ID}' already exists")
    except RepositoryNotFoundError:
        print(f"🆕 Creating new space '{REPO_ID}'...")
        try:
            create_repo(
                repo_id=REPO_ID,
                repo_type="space",
                space_sdk="docker",
                private=False
            )
            print("✅ Space created successfully")
        except Exception as e:
            print(f"❌ Error creating space: {e}")
            return False
    
    # Verify deployment files exist
    deployment_dir = Path("tourism_mlops/deployment")
    required_files = ["Dockerfile", "app.py", "requirements.txt"]
    
    print("\n🔍 Verifying deployment files...")
    for file_name in required_files:
        file_path = deployment_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    # Create README.md for the space
    readme_content = """---
title: Tourism Package Prediction
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
license: mit
---

# Tourism Package Prediction 🏖️

An intelligent ML-powered system that predicts tourism package purchase likelihood based on customer behavior and demographics.

## Features
- **Real-time Predictions**: Instant customer scoring using trained ML models
- **Interactive Interface**: User-friendly Streamlit web application
- **Multiple Algorithms**: Best model selected from 6 different algorithms
- **MLOps Integration**: Complete pipeline with experiment tracking

## How to Use
1. Enter customer demographic information
2. Provide interaction and financial details
3. Get instant purchase likelihood prediction
4. View confidence scores and recommendations

## Technical Stack
- **Backend**: Python, Scikit-learn, XGBoost
- **Frontend**: Streamlit
- **ML Tracking**: MLflow
- **Deployment**: Docker, HuggingFace Spaces
- **Model Storage**: HuggingFace Hub

## Model Performance
The system uses the best-performing model from a comprehensive comparison of:
- Decision Tree
- Random Forest  
- Bagging Classifier
- AdaBoost
- Gradient Boosting
- XGBoost

Built with ❤️ using MLOps best practices.
"""
    
    # Write README.md
    readme_path = deployment_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print("✅ README.md created")
    
    # Upload files to HuggingFace Space
    print("\n📤 Uploading files to HuggingFace Space...")
    try:
        api.upload_folder(
            folder_path=str(deployment_dir),
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Deploy Tourism Package Prediction application"
        )
        print("✅ All files uploaded successfully")
        
        # Generate space URL
        space_url = f"https://huggingface.co/spaces/{REPO_ID}"
        print(f"\n🎉 Deployment completed successfully!")
        print(f"🌐 Your application is available at: {space_url}")
        print(f"⏱️  It may take a few minutes for the space to build and become available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 TOURISM PACKAGE PREDICTION DEPLOYMENT")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("HF_TOKEN"):
        print("❌ HF_TOKEN environment variable not set!")
        print("💡 Please set your HuggingFace token:")
        print("   export HF_TOKEN='your_token_here'")
        return
    
    # Execute deployment
    success = deploy_to_huggingface_space()
    
    if success:
        print("\n✅ Deployment process completed successfully!")
        print("📱 Your Tourism Package Prediction app is now live on HuggingFace Spaces!")
    else:
        print("\n❌ Deployment failed. Please check the errors above.")

if __name__ == "__main__":
    main()
