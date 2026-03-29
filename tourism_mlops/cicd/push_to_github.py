
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(command, cwd=None):
    """Execute shell command and return result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {str(e)}")
        return "", str(e), 1

def check_git_status():
    """Check current Git status"""
    print("🔍 Checking Git status...")
    
    stdout, stderr, code = run_command("git status --porcelain")
    if code != 0:
        print(f"❌ Failed to check git status: {stderr}")
        return False
    
    if stdout:
        print("📝 Files to be committed:")
        for line in stdout.split('\\n'):
            if line.strip():
                print(f"   {line}")
        return True
    else:
        print("ℹ️ Working directory clean - no changes to commit")
        return False

def stage_all_files():
    """Stage all files for Git commit"""
    print("📤 Staging all files...")
    
    stdout, stderr, code = run_command("git add .")
    if code != 0:
        print(f"❌ Failed to stage files: {stderr}")
        return False
    
    print("✅ All files staged successfully")
    return True

def create_commit(message=None):
    """Create Git commit with message"""
    if not message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚀 MLOps Pipeline Update - {timestamp}"
    
    print(f"💾 Creating commit: {message}")
    
    stdout, stderr, code = run_command(f'git commit -m "{message}"')
    if code != 0:
        if "nothing to commit" in stderr:
            print("ℹ️ Nothing to commit - working directory clean")
            return True
        print(f"❌ Failed to create commit: {stderr}")
        return False
    
    print("✅ Commit created successfully")
    return True

def push_to_remote(branch="main"):
    """Push commits to remote repository"""
    print(f"🚀 Pushing to remote branch: {branch}")
    
    # First, try to push
    stdout, stderr, code = run_command(f"git push origin {branch}")
    
    if code != 0:
        if "failed to push some refs" in stderr:
            print("⚠️ Push rejected. Trying to pull and merge...")
            
            # Pull with rebase
            stdout_pull, stderr_pull, code_pull = run_command(f"git pull --rebase origin {branch}")
            if code_pull != 0:
                print(f"❌ Failed to pull and rebase: {stderr_pull}")
                return False
            
            # Try pushing again
            stdout, stderr, code = run_command(f"git push origin {branch}")
    
    if code != 0:
        print(f"❌ Failed to push: {stderr}")
        return False
    
    print("✅ Successfully pushed to remote repository")
    return True

def setup_branch(branch_name="main"):
    """Setup and switch to specified branch"""
    print(f"🌿 Setting up branch: {branch_name}")
    
    # Check current branch
    stdout, stderr, code = run_command("git branch --show-current")
    current_branch = stdout.strip() if code == 0 else "unknown"
    
    if current_branch == branch_name:
        print(f"ℹ️ Already on branch: {branch_name}")
        return True
    
    # Check if branch exists
    stdout, stderr, code = run_command(f"git show-ref --verify --quiet refs/heads/{branch_name}", check=False)
    
    if code == 0:
        # Branch exists, switch to it
        stdout, stderr, code = run_command(f"git checkout {branch_name}")
    else:
        # Branch doesn't exist, create and switch
        stdout, stderr, code = run_command(f"git checkout -b {branch_name}")
    
    if code != 0:
        print(f"❌ Failed to setup branch: {stderr}")
        return False
    
    print(f"✅ Successfully setup branch: {branch_name}")
    return True

def verify_repository():
    """Verify Git repository is properly configured"""
    print("🔧 Verifying repository configuration...")
    
    # Check if we're in a git repository
    stdout, stderr, code = run_command("git rev-parse --is-inside-work-tree")
    if code != 0:
        print("❌ Not inside a Git repository")
        return False
    
    # Check if remote origin exists
    stdout, stderr, code = run_command("git remote get-url origin")
    if code != 0:
        print("⚠️ No remote origin configured")
        return False
    else:
        print(f"✅ Remote origin: {stdout}")
    
    # Check Git user configuration
    stdout_name, stderr_name, code_name = run_command("git config user.name")
    stdout_email, stderr_email, code_email = run_command("git config user.email")
    
    if code_name != 0 or code_email != 0:
        print("⚠️ Git user not configured, setting defaults...")
        run_command('git config user.name "sdj206gl"')
        run_command('git config user.email "shashidhar.jagatap206@gmail.com"')
    else:
        print(f"✅ Git user: {stdout_name} <{stdout_email}>")
    
    return True

def create_workflow_summary():
    """Create a workflow execution summary"""
    summary = f"""
# 🤖 MLOps Workflow Execution Summary

**Execution Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 📋 Workflow Steps Completed:
- ✅ Repository verification
- ✅ File staging  
- ✅ Commit creation
- ✅ Branch management
- ✅ Remote push

## 🚀 GitHub Actions Pipeline:
The following pipeline will be triggered automatically:
1. **Code Quality**: Linting, formatting, testing
2. **Data Pipeline**: Registration and preparation
3. **Model Training**: Multi-algorithm comparison
4. **Model Validation**: Performance thresholds
5. **Deployment**: HuggingFace Spaces
6. **Monitoring**: Performance tracking
7. **Auto-update**: Main branch updates

## 📊 Expected Outcomes:
- Models trained and compared
- Best model registered to HuggingFace Hub
- Application deployed to HuggingFace Spaces
- Performance report generated
- Repository automatically updated

## 🔗 Resources:
- **Repository**: GitHub repository with latest code
- **CI/CD Pipeline**: GitHub Actions workflow
- **Application**: HuggingFace Spaces deployment
- **Models**: HuggingFace Hub model registry

---
*Generated by Tourism Package Prediction MLOps Pipeline*
"""
    
    # Save summary
    with open('workflow_summary.md', 'w') as f:
        f.write(summary)
    
    print("📄 Workflow summary created: workflow_summary.md")

def main():
    """Main execution function for pushing to GitHub"""
    print("=" * 60)
    print("🚀 PUSH TO GITHUB - AUTOMATED WORKFLOW")
    print("=" * 60)
    
    # Verify repository setup
    if not verify_repository():
        print("❌ Repository verification failed. Run setup_repository.py first.")
        return False
    
    # Setup main branch
    if not setup_branch("main"):
        return False
    
    # Check what needs to be committed
    changes_exist = check_git_status()
    
    # Create workflow summary
    create_workflow_summary()
    
    # Stage files if changes exist
    if changes_exist or os.path.exists('workflow_summary.md'):
        if not stage_all_files():
            return False
        
        # Create commit
        commit_message = f"🚀 Complete MLOps Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not create_commit(commit_message):
            return False
        
        # Push to remote
        if not push_to_remote("main"):
            return False
    
    print("\\n" + "=" * 60)
    print("🎉 Successfully pushed to GitHub!")
    print("📋 What happens next:")
    print("   1. GitHub Actions pipeline will be triggered")
    print("   2. Code quality checks will run")
    print("   3. Data processing will execute")  
    print("   4. Models will be trained and compared")
    print("   5. Best model will be validated and registered")
    print("   6. Application will be deployed to HuggingFace Spaces")
    print("   7. Performance report will be generated")
    print("   8. Repository will be automatically updated")
    print("\\n🌐 Monitor progress at:")
    print("   - GitHub Actions: https://github.com/sdj206gl/Tourism_Package_Prediction/actions")
    print("   - HuggingFace Spaces: https://huggingface.co/spaces/shashidj/Tourism-Package-Prediction")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
