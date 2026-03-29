
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Execute shell command and return result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True,
            check=check
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e.stderr}")
        return "", e.stderr, e.returncode

def initialize_git_repository():
    """Initialize Git repository and configure settings"""
    print("🔧 Initializing Git repository...")
    
    # Initialize git if not already done
    if not os.path.exists('.git'):
        stdout, stderr, code = run_command("git init")
        if code != 0:
            print(f"❌ Failed to initialize git: {stderr}")
            return False
        print("✅ Git repository initialized")
    else:
        print("ℹ️ Git repository already exists")
    
    # Configure git user (if not already configured)
    run_command("git config --global user.name 'sdj206gl' || true", check=False)
    run_command("git config --global user.email 'shashidhar.jagatap206@gmail.com' || true", check=False)
    
    return True

def create_gitignore():
    """Create comprehensive .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
pip-log.txt
pip-delete-this-directory.txt

# Data files
*.csv
*.json
*.pkl
*.h5
*.hdf5
tourism_mlops/data/

# ML and DS
mlruns/
.mlflow/
*.model
*.bin
experiments/
logs/
checkpoints/

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment variables
.env
.env.local

# Docker
.dockerignore
docker-compose.override.yml

# Temporary files
*.tmp
*.temp
temp/
tmp/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    
    print("✅ .gitignore file created")

def create_github_repository():
    """Create GitHub repository using GitHub CLI"""
    repo_name = "Tourism_Package_Prediction"
    
    print(f"🚀 Creating GitHub repository: {repo_name}")
    
    # Check if GitHub CLI is installed
    stdout, stderr, code = run_command("gh --version", check=False)
    if code != 0:
        print("⚠️ GitHub CLI not found. Please install it:")
        print("   🍎 macOS: brew install gh")
        print("   🐧 Ubuntu: sudo apt install gh")
        print("   🪟 Windows: winget install GitHub.cli")
        print("\nAlternatively, create the repository manually at: https://github.com/new")
        return False
    
    # Create repository
    create_cmd = f"gh repo create {repo_name} --public --description 'Tourism Package Prediction MLOps Pipeline' --clone=false"
    stdout, stderr, code = run_command(create_cmd, check=False)
    
    if code != 0 and "already exists" not in stderr:
        print(f"❌ Failed to create repository: {stderr}")
        print("💡 You may need to authenticate with: gh auth login")
        return False
    elif "already exists" in stderr:
        print("ℹ️ Repository already exists")
    else:
        print("✅ GitHub repository created successfully")
    
    return True

def setup_remote_origin():
    """Setup remote origin for GitHub repository"""
    print("🔗 Setting up remote origin...")
    
    # Get GitHub username
    stdout, stderr, code = run_command("gh api user --jq '.login'", check=False)
    if code != 0:
        print("⚠️ Could not get GitHub username. Using placeholder.")
        username = "sdj206gl"
    else:
        username = stdout.strip()
    
    repo_url = f"https://github.com/{username}/Tourism_Package_Prediction.git"
    
    # Add remote origin
    run_command("git remote remove origin", check=False)  # Remove if exists
    stdout, stderr, code = run_command(f"git remote add origin {repo_url}")
    
    if code != 0:
        print(f"❌ Failed to add remote origin: {stderr}")
        return False
    
    print(f"✅ Remote origin set to: {repo_url}")
    return True

def create_directory_structure():
    """Create complete project directory structure"""
    print("📁 Creating project directory structure...")
    
    directories = [
        "tourism_mlops/data",
        "tourism_mlops/model_building",
        "tourism_mlops/deployment",
        "tourism_mlops/cicd",
        ".github/workflows",
        "reports",
        "docs",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py for Python packages
        if "tourism_mlops" in directory and directory != "tourism_mlops/data":
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Tourism Package Prediction MLOps\\n")
    
    print("✅ Directory structure created")

def create_readme():
    """Create comprehensive README.md"""
    readme_content = """# Tourism Package Prediction MLOps Pipeline 🏖️

An end-to-end machine learning operations (MLOps) pipeline for predicting tourism package purchases using customer behavior data.

## 🎯 Project Overview

This project implements a complete MLOps workflow including:
- **Data Processing**: Automated data ingestion and preprocessing
- **Model Training**: Multi-algorithm comparison with hyperparameter tuning
- **Model Registry**: HuggingFace Hub integration for model versioning
- **Deployment**: Containerized deployment to HuggingFace Spaces
- **CI/CD Pipeline**: GitHub Actions workflow for automation
- **Monitoring**: Performance tracking and automated reporting

## 🧠 Machine Learning Models

The pipeline trains and compares 6 different algorithms:
- Decision Tree
- Random Forest
- Bagging Classifier
- AdaBoost
- Gradient Boosting
- XGBoost

## 🚀 MLOps Pipeline

### Data Pipeline
1. **Data Registration**: Upload dataset to HuggingFace Hub
2. **Data Preparation**: Clean, encode, and split data
3. **Feature Engineering**: Transform features for ML models

### Model Pipeline
1. **Training**: Train multiple models with hyperparameter tuning
2. **Evaluation**: Compare models using multiple metrics
3. **Registration**: Register best model to HuggingFace Hub

### Deployment Pipeline
1. **Containerization**: Docker-based deployment
2. **Web Application**: Streamlit interface for predictions
3. **Cloud Deployment**: HuggingFace Spaces hosting

### CI/CD Pipeline
- **Code Quality**: Linting, formatting, and testing
- **Automated Training**: Trigger on code changes
- **Model Validation**: Performance threshold checks
- **Automated Deployment**: Deploy on successful validation

## 🏭 Infrastructure

- **Version Control**: Git with automated commits
- **Container Registry**: Docker Hub integration
- **Model Registry**: HuggingFace Hub
- **Deployment Platform**: HuggingFace Spaces
- **CI/CD**: GitHub Actions
- **Monitoring**: MLflow experiment tracking

## 📊 Model Performance

The pipeline automatically selects the best-performing model based on:
- Accuracy
- F1-Score
- ROC-AUC
- Cross-validation performance

## 🚀 Quick Start

### Prerequisites
```bash
# Install required tools
brew install gh  # GitHub CLI (macOS)
pip install -r tourism_mlops/deployment/requirements.txt
```

### Setup
```bash
# Clone repository
git clone https://github.com/sdj206gl/Tourism_Package_Prediction.git
cd Tourism_Package_Prediction

# Set environment variables
export HF_TOKEN="your_huggingface_token_here"

# Initialize pipeline
python tourism_mlops/cicd/setup_repository.py
```

### Manual Execution
```bash
# Data processing
python tourism_mlops/model_building/data_register.py
python tourism_mlops/model_building/prep.py

# Model training
python tourism_mlops/model_building/model_training.py

# Deployment
python tourism_mlops/deployment/deploy_to_hf_space.py
```

## 📁 Project Structure

```
Tourism_Package_Prediction/
├── .github/workflows/           # GitHub Actions CI/CD
│   └── pipeline.yml
├── tourism_mlops/
│   ├── data/                    # Dataset storage
│   ├── model_building/          # ML pipeline scripts
│   ├── deployment/              # Deployment configurations  
│   └── cicd/                    # CI/CD utilities
├── reports/                     # Performance reports
├── docs/                        # Documentation
└── tests/                       # Test scripts
```

## 🔗 Links

- **Live Application**: [HuggingFace Spaces](https://huggingface.co/spaces/shashidj/Tourism-Package-Prediction)
- **Model Registry**: [HuggingFace Hub](https://huggingface.co/shashidj/tourism-package-prediction-model)
- **Dataset**: [HuggingFace Datasets](https://huggingface.co/datasets/shashidj/tourism-package-prediction)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🏷️ Tags

`machine-learning` `mlops` `tourism` `prediction` `ci-cd` `docker` `streamlit` `huggingface` `github-actions`
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md created")

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 TOURISM PACKAGE PREDICTION - REPOSITORY SETUP")
    print("=" * 60)
    
    # Initialize repository
    if not initialize_git_repository():
        return False
    
    # Create project structure
    create_directory_structure()
    create_gitignore()
    create_readme()
    
    # GitHub repository setup
    if create_github_repository():
        setup_remote_origin()
    
    print("\n" + "=" * 60)
    print("✅ Repository setup completed successfully!")
    print("📋 Next Steps:")
    print("   1. Set HF_TOKEN environment variable")
    print("   2. Upload tourism.csv to tourism_mlops/data/")
    print("   3. Run: git add . && git commit -m 'Initial commit'")
    print("   4. Run: git push -u origin main")
    print("   5. Enable GitHub Actions in repository settings")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
