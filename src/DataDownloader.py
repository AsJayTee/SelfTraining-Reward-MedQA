"""
DataDownloader.py

Downloads medical Q&A repositories for the DSA4213 project and tracks their versions.
Downloads:
- MedQuAD dataset from https://github.com/abachaa/MedQuAD
- LiveQA Medical Task test questions from https://github.com/abachaa/LiveQA_MedicalTask_TREC2017

Author: DSA4213 Group Project Team 18
"""

import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """
    Downloads and manages medical Q&A datasets for the project.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DataDownloader.
        
        Args:
            data_dir: Directory where repositories will be downloaded
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.repositories = {
            "MedQuAD": {
                "url": "https://github.com/abachaa/MedQuAD.git",
                "folder": "MedQuAD",
                "description": "Medical Question Answering Dataset - 47,457 medical QA pairs from NIH websites"
            },
            "TestQuestions": {
                "url": "https://github.com/abachaa/LiveQA_MedicalTask_TREC2017.git",
                "folder": "TestQuestions", 
                "description": "LiveQA Medical Task TREC 2017 test questions and evaluation data"
            }
        }
        
        self.version_file = self.data_dir / "repository_versions.json"
    
    def run_git_command(self, command: list, cwd: Optional[Path] = None) -> tuple[bool, str]:
        """
        Run a git command and return success status and output.
        
        Args:
            command: Git command as list of strings
            cwd: Working directory for the command
            
        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(command)}")
            logger.error(f"Error: {e.stderr}")
            return False, e.stderr
        except FileNotFoundError:
            logger.error("Git is not installed or not in PATH")
            return False, "Git not found"
    
    def get_repository_info(self, repo_path: Path) -> Dict[str, str]:
        """
        Get version information from a git repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary with repository information
        """
        info = {}
        
        # Get current commit hash
        success, commit_hash = self.run_git_command(
            ["git", "rev-parse", "HEAD"], 
            cwd=repo_path
        )
        if success:
            info["commit_hash"] = commit_hash
            info["short_hash"] = commit_hash[:8]
        
        # Get commit date
        success, commit_date = self.run_git_command(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_path
        )
        if success:
            info["commit_date"] = commit_date
        
        # Get branch name
        success, branch = self.run_git_command(
            ["git", "branch", "--show-current"],
            cwd=repo_path
        )
        if success:
            info["branch"] = branch if branch else "main"
        
        # Get remote origin URL
        success, remote_url = self.run_git_command(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path
        )
        if success:
            info["remote_url"] = remote_url
            
        return info
    
    def clone_repository(self, repo_name: str) -> bool:
        """
        Clone a repository if it doesn't exist, otherwise pull latest changes.
        
        Args:
            repo_name: Name of the repository (key in self.repositories)
            
        Returns:
            True if successful, False otherwise
        """
        repo_config = self.repositories[repo_name]
        repo_path = self.data_dir / repo_config["folder"]
        
        logger.info(f"Processing repository: {repo_name}")
        logger.info(f"Description: {repo_config['description']}")
        
        if repo_path.exists() and (repo_path / ".git").exists():
            logger.info(f"Repository already exists at {repo_path}")
            logger.info("Pulling latest changes...")
            
            # Pull latest changes
            success, output = self.run_git_command(
                ["git", "pull", "origin", "main"],
                cwd=repo_path
            )
            
            if not success:
                # Try 'master' branch if 'main' doesn't work
                logger.info("Trying master branch...")
                success, output = self.run_git_command(
                    ["git", "pull", "origin", "master"],
                    cwd=repo_path
                )
            
            if success:
                logger.info("Repository updated successfully")
                return True
            else:
                logger.error(f"Failed to update repository: {output}")
                return False
        
        else:
            logger.info(f"Cloning repository to {repo_path}")
            
            # Clone the repository
            success, output = self.run_git_command([
                "git", "clone", 
                repo_config["url"], 
                str(repo_path)
            ])
            
            if success:
                logger.info("Repository cloned successfully")
                return True
            else:
                logger.error(f"Failed to clone repository: {output}")
                return False
    
    def update_version_tracking(self):
        """
        Update the version tracking file with current repository information.
        """
        version_info = {
            "last_updated": datetime.now().isoformat(),
            "repositories": {}
        }
        
        for repo_name, repo_config in self.repositories.items():
            repo_path = self.data_dir / repo_config["folder"]
            
            if repo_path.exists() and (repo_path / ".git").exists():
                repo_info = self.get_repository_info(repo_path)
                repo_info.update({
                    "folder_name": repo_config["folder"],
                    "description": repo_config["description"],
                    "download_status": "success"
                })
                version_info["repositories"][repo_name] = repo_info
            else:
                version_info["repositories"][repo_name] = {
                    "folder_name": repo_config["folder"],
                    "description": repo_config["description"],
                    "download_status": "failed"
                }
        
        # Save version information
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Version information saved to {self.version_file}")
    
    def display_version_info(self):
        """
        Display current version information.
        """
        if not self.version_file.exists():
            logger.warning("No version information found. Run download_all() first.")
            return
        
        with open(self.version_file, 'r') as f:
            version_info = json.load(f)
        
        print("\n" + "="*60)
        print("REPOSITORY VERSION INFORMATION")
        print("="*60)
        print(f"Last Updated: {version_info['last_updated']}")
        print()
        
        for repo_name, repo_info in version_info["repositories"].items():
            print(f"Repository: {repo_name}")
            print(f"  Folder: data/raw/{repo_info['folder_name']}")
            print(f"  Description: {repo_info['description']}")
            print(f"  Status: {repo_info['download_status']}")
            
            if repo_info["download_status"] == "success":
                print(f"  Commit: {repo_info.get('short_hash', 'Unknown')} ({repo_info.get('commit_date', 'Unknown')})")
                print(f"  Branch: {repo_info.get('branch', 'Unknown')}")
                print(f"  Remote: {repo_info.get('remote_url', 'Unknown')}")
            
            print()
    
    def get_dataset_statistics(self) -> Dict[str, Dict]:
        """
        Get basic statistics about the downloaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        # MedQuAD statistics
        medquad_path = self.data_dir / "MedQuAD"
        if medquad_path.exists():
            try:
                # Count XML files in different directories
                qa_files = list(medquad_path.rglob("*.xml"))
                stats["MedQuAD"] = {
                    "total_files": len(qa_files),
                    "size_mb": sum(f.stat().st_size for f in qa_files) / (1024 * 1024),
                    "folders": [d.name for d in medquad_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                }
            except Exception as e:
                stats["MedQuAD"] = {"error": str(e)}
        
        # TestQuestions statistics  
        testq_path = self.data_dir / "TestQuestions"
        if testq_path.exists():
            try:
                # Count different file types
                all_files = list(testq_path.rglob("*"))
                files_only = [f for f in all_files if f.is_file()]
                stats["TestQuestions"] = {
                    "total_files": len(files_only),
                    "size_mb": sum(f.stat().st_size for f in files_only) / (1024 * 1024),
                    "file_types": list(set(f.suffix for f in files_only if f.suffix))
                }
            except Exception as e:
                stats["TestQuestions"] = {"error": str(e)}
        
        return stats
    
    def download_all(self) -> bool:
        """
        Download all configured repositories.
        
        Returns:
            True if all downloads were successful, False otherwise
        """
        logger.info("Starting download of all repositories...")
        
        success_count = 0
        total_count = len(self.repositories)
        
        for repo_name in self.repositories:
            if self.clone_repository(repo_name):
                success_count += 1
                logger.info(f"✓ Successfully processed {repo_name}")
            else:
                logger.error(f"✗ Failed to process {repo_name}")
            print()  # Add spacing between repositories
        
        # Update version tracking
        self.update_version_tracking()
        
        # Display results
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Successfully processed: {success_count}/{total_count} repositories")
        
        if success_count == total_count:
            logger.info("All repositories downloaded successfully!")
            
            # Show dataset statistics
            stats = self.get_dataset_statistics()
            if stats:
                print("\nDATASET STATISTICS:")
                print("-" * 30)
                for dataset, info in stats.items():
                    if "error" not in info:
                        print(f"{dataset}:")
                        print(f"  Files: {info['total_files']}")
                        print(f"  Size: {info['size_mb']:.1f} MB")
                        if "folders" in info:
                            print(f"  Folders: {', '.join(info['folders'][:5])}{'...' if len(info['folders']) > 5 else ''}")
                        if "file_types" in info:
                            print(f"  File types: {', '.join(info['file_types'])}")
                        print()
            
            return True
        else:
            logger.error("Some repositories failed to download")
            return False


def main():
    """
    Main function to run the data downloader.
    """
    print("Medical Q&A Data Downloader")
    print("DSA4213 Group Project")
    print("="*50)
    
    downloader = DataDownloader()
    
    # Download all repositories
    success = downloader.download_all()
    
    # Display version information
    downloader.display_version_info()
    
    if success:
        print("✓ All data ready for the medical Q&A project!")
        print("\nNext steps:")
        print("1. Explore data/raw/MedQuAD/ for the main medical Q&A dataset")
        print("2. Check data/raw/TestQuestions/ for evaluation questions")
        print("3. Review data/raw/repository_versions.json for version tracking")
    else:
        print("✗ Some downloads failed. Check the logs above for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())