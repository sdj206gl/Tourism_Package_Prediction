
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

def run_command(command, cwd=None):
    """Execute shell command with real-time output"""
    print(f"🔄 Executing: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(f"   {line.rstrip()}")
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    python_version = sys.version
    print(f"✅ Python version: {python_version}")
    
    # Check required environment variables
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set")
        print("💡 Set it with: export HF_TOKEN='your_huggingface_token_here'")
        return False
    else:
        print("✅ HF_TOKEN environment variable found")
    
    # Check if Git is installed
    if not run_command("git --version"):
        print("❌ Git is not installed")
        return False
    print("✅ Git is available")
    
    # Check if we have the required files
    required_files = [
        ".github/workflows/pipeline.yml",
        "tourism_mlops/model_building/data_register.py",
        "tourism_mlops/model_building/prep.py", 
        "tourism_mlops/model_building/model_training.py",
        "tourism_mlops/deployment/Dockerfile",
        "tourism_mlops/deployment/app.py",
        "tourism_mlops/deployment/requirements.txt",
        "tourism_mlops/deployment/deploy_to_hf_space.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All required files present")
    
    return True

def setup_environment():
    """Setup the development environment"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        "tourism_mlops/data",
        "reports", 
        "docs",
        "tests",
        ".github/workflows"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def execute_local_pipeline():
    """Execute the ML pipeline locally for testing"""
    print("🚀 Executing local ML pipeline...")
    
    # Step 1: Data Registration
    print("\\n📊 Step 1: Data Registration")
    if not run_command("python data_register.py", cwd="tourism_mlops/model_building"):
        print("❌ Data registration failed")
        return False
    
    # Step 2: Data Preparation
    print("\\n🔄 Step 2: Data Preparation")
    if not run_command("python prep.py", cwd="tourism_mlops/model_building"):
        print("❌ Data preparation failed") 
        return False
    
    # Step 3: Model Training (Optional - comment out for faster testing)
    print("\\n🤖 Step 3: Model Training (This may take several minutes...)")
    print("💡 Skipping model training in automation script - will be done by GitHub Actions")
    # if not run_command("python model_training.py", cwd="tourism_mlops/model_building"):
    #     print("❌ Model training failed")
    #     return False
    
    print("✅ Local pipeline executed successfully")
    return True

def initialize_repository():
    """Initialize Git repository"""
    print("📁 Initializing repository...")
    
    # Run repository setup script
    if not run_command("python setup_repository.py", cwd="tourism_mlops/cicd"):
        print("❌ Repository setup failed")
        return False
    
    print("✅ Repository initialized successfully")
    return True

def push_to_github():
    """Push all files to GitHub"""
    print("📤 Pushing to GitHub...")
    
    # Run GitHub push script
    if not run_command("python push_to_github.py", cwd="tourism_mlops/cicd"):
        print("❌ GitHub push failed")
        return False
    
    print("✅ Successfully pushed to GitHub")
    return True

def monitor_github_actions():
    """Monitor GitHub Actions workflow"""
    print("👀 Monitoring GitHub Actions...")
    print("🌐 Check workflow status at:")
    print("   https://github.com/sdj206gl/Tourism_Package_Prediction/actions")
    print("\\n⏱️  The workflow includes the following steps:")
    print("   1. Code Quality & Testing")
    print("   2. Data Registration & Preparation")  
    print("   3. Model Training & Hyperparameter Tuning")
    print("   4. Model Testing & Validation")
    print("   5. Deploy to HuggingFace Spaces") 
    print("   6. Performance Monitoring & Reporting")
    print("   7. Auto-update Main Branch")
    print("   8. Cleanup")
    print("\\n📊 Expected completion time: 15-30 minutes")

def create_automation_report():
    """Create automation execution report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    report_content = f"""# 🤖 MLOps Automation Execution Report

**Execution Timestamp**: {timestamp}

## ✅ Completed Steps

### 1. Prerequisites Check
- Python environment: ✅ Verified
- Environment variables: ✅ HF_TOKEN configured
- Required files: ✅ All present
- Git installation: ✅ Available

### 2. Environment Setup
- Directory structure: ✅ Created
- Configuration files: ✅ Ready

### 3. Local Pipeline Execution
- Data registration: ✅ Completed
- Data preparation: ✅ Completed
- Model training: ⏭️ Delegated to GitHub Actions

### 4. Repository Management
- Git repository: ✅ Initialized
- GitHub repository: ✅ Created/Updated
- Remote configuration: ✅ Set

### 5. GitHub Integration
- Files pushed: ✅ All MLOps components
- GitHub Actions: 🚀 Triggered automatically
- Workflow monitoring: 📊 Available

## 🚀 GitHub Actions Pipeline Status

The following workflow will execute automatically:

| Stage | Description | Expected Duration |
|-------|-------------|-------------------|
| Code Quality | Linting, formatting, testing | 2-3 minutes |
| Data Pipeline | Registration & preparation | 3-5 minutes |
| Model Training | 6 algorithms with tuning | 10-15 minutes |
| Model Validation | Performance testing | 2-3 minutes |
| Deployment | HuggingFace Spaces | 3-5 minutes |
| Monitoring | Performance tracking | 1-2 minutes |
| Auto-update | Branch management | 1 minute |

**Total Estimated Time**: 25-35 minutes

## 🔗 Resources Generated

- **GitHub Repository**: https://github.com/sdj206gl/Tourism_Package_Prediction
- **GitHub Actions**: https://github.com/sdj206gl/Tourism_Package_Prediction/actions
- **HuggingFace Space** (after deployment): https://huggingface.co/spaces/shashidj/Tourism-Package-Prediction
- **Model Registry** (after training): https://huggingface.co/shashidj/tourism-package-prediction-model
- **Dataset Registry**: https://huggingface.co/datasets/shashidj/tourism-package-prediction

## 📊 Next Steps

1. **Monitor GitHub Actions**: Check workflow progress in the Actions tab
2. **Review Logs**: Examine each job's execution logs for any issues
3. **Verify Deployment**: Test the deployed application once workflow completes
4. **Check Model Registry**: Confirm best model is registered
5. **Performance Review**: Examine the generated performance report

## 🎯 Success Criteria

- ✅ All GitHub Actions jobs complete successfully
- ✅ Application deploys to HuggingFace Spaces
- ✅ Best model registers to HuggingFace Hub  
- ✅ Performance report generates with metrics
- ✅ Repository updates automatically

---
*Generated by Tourism Package Prediction MLOps Automation Pipeline*
"""
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/automation_report.md", "w") as f:
        f.write(report_content)
    
    print("📄 Automation report saved: reports/automation_report.md")

def main():
    """Main automation workflow"""
    print("=" * 70)
    print("🚀 TOURISM PACKAGE PREDICTION - COMPLETE MLOPS AUTOMATION")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Step 1: Check Prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please resolve issues and try again.")
        return False
    
    # Step 2: Setup Environment  
    if not setup_environment():
        print("❌ Environment setup failed.")
        return False
    
    # Step 3: Execute Local Pipeline (Data processing only)
    if not execute_local_pipeline():
        print("❌ Local pipeline execution failed.")
        return False
    
    # Step 4: Initialize Repository
    if not initialize_repository():
        print("❌ Repository initialization failed.")
        return False
    
    # Step 5: Push to GitHub
    if not push_to_github():
        print("❌ GitHub push failed.")
        return False
    
    # Step 6: Create Report
    create_automation_report()
    
    # Step 7: Monitor GitHub Actions
    monitor_github_actions()
    
    print("\\n" + "=" * 70)
    print("🎉 MLOps AUTOMATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("📋 Summary:")
    print("   ✅ Prerequisites verified")
    print("   ✅ Environment configured") 
    print("   ✅ Local data pipeline executed")
    print("   ✅ Repository initialized")
    print("   ✅ Code pushed to GitHub")
    print("   ✅ GitHub Actions triggered")
    print("   ✅ Automation report generated")
    print()
    print("🚀 The end-to-end MLOps pipeline is now running automatically!")
    print("📊 Monitor progress at: https://github.com/sdj206gl/Tourism_Package_Prediction/actions")
    print()
    print("⏰ Estimated completion: 25-35 minutes")
    print("🎯 Final deliverables will include:")
    print("   - Trained and validated ML models")
    print("   - Deployed web application")
    print("   - Performance monitoring report")
    print("   - Automated repository updates")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
