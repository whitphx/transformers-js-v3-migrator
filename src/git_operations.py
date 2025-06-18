import os
import shutil
import tempfile
from typing import Optional, List
from huggingface_hub import HfApi, CommitOperationAdd, snapshot_download
import logging


class GitOperations:
    def __init__(self, token: Optional[str] = None, verbose: bool = False):
        self.logger = logging.getLogger(__name__)
        self.repo_path = None
        self.token = token
        self.verbose = verbose
        self.hf_api = HfApi(token=token)

    def download_repo(self, repo_id: str) -> str:
        """Download a repository from Hugging Face Hub to a temporary working directory"""
        import tempfile
        import shutil
        
        try:
            self.logger.info(f"Downloading {repo_id}")
            
            # Use HF Hub's snapshot_download to get repository files (cached)
            cached_repo_path = snapshot_download(
                repo_id=repo_id,
                token=self.token,
                repo_type="model"
            )
            
            # Create a temporary working directory
            temp_dir = tempfile.mkdtemp(prefix=f"transformersjs_migration_{repo_id.replace('/', '_')}_")
            
            # Copy cached repository to temporary directory for manipulation
            for item in os.listdir(cached_repo_path):
                src_path = os.path.join(cached_repo_path, item)
                dst_path = os.path.join(temp_dir, item)
                
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
            
            self.repo_path = temp_dir
            self.logger.info(f"Downloaded {repo_id} to cache and copied to working directory: {temp_dir}")
            return temp_dir
        except Exception as e:
            self.logger.error(f"Failed to download {repo_id}: {e}")
            raise

    def upload_changes(self, repo_path: str, repo_id: str, migration_type_name: str, 
                      migration_title: str, migration_description: str, 
                      modified_files: List[str], interactive: bool = True) -> Optional[str]:
        """Upload changes to the repository using HF Hub API
        
        Returns:
            str: PR URL if successful, None if failed
        """
        try:
            if not modified_files:
                self.logger.info(f"No files to upload for {repo_id}")
                return None
            
            
            # Prepare commit operations for modified files
            operations = []
            for file_path in modified_files:
                local_file_path = os.path.join(repo_path, file_path)
                if os.path.exists(local_file_path):
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=file_path,
                            path_or_fileobj=local_file_path
                        )
                    )
                    self.logger.info(f"Prepared upload for {file_path}")
            
            if not operations:
                self.logger.info(f"No valid files to upload for {repo_id}")
                return None
            
            # Create simple commit message
            commit_message = migration_title
            
            # Preview PR content if in interactive mode
            if interactive:
                self._preview_pr_content(repo_id, migration_title, migration_description, files_modified)
                
                # Ask for user confirmation
                response = input("\n🤔 Do you want to create this pull request? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    self.logger.info("Pull request creation cancelled by user")
                    return None
            
            # Create commit and auto-create PR (HF Hub will create branch automatically)
            self.logger.info(f"Creating commit with PR for {migration_type_name} migration")
            
            commit_info = self.hf_api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                create_pr=True,  # Auto-create PR with HF-generated branch name
                repo_type="model"
                # No revision specified - HF Hub creates branch automatically
            )
            
            self.logger.info(f"Successfully uploaded changes to {repo_id}")
            self.logger.info(f"Commit URL: {commit_info.commit_url}")
            
            # Update PR with custom title and description if PR was created
            pr_url = None
            if hasattr(commit_info, 'pr_url') and commit_info.pr_url and hasattr(commit_info, 'pr_num'):
                pr_url = commit_info.pr_url
                pr_num = commit_info.pr_num
                self.logger.info(f"Auto-created pull request: {pr_url}")
                
                # Update PR description (title is automatically set from commit message)
                try:
                    # Get discussion details to find the first comment ID (PR description)
                    discussion_details = self.hf_api.get_discussion_details(
                        repo_id=repo_id,
                        discussion_num=pr_num,
                        repo_type="model"
                    )
                    
                    # Find the first comment (PR description) - it should be the first event
                    first_comment_id = None
                    for event in discussion_details.events:
                        if hasattr(event, 'id') and hasattr(event, 'type') and event.type == 'comment':
                            first_comment_id = event.id
                            break
                    
                    if first_comment_id:
                        # Update PR description by editing the first comment
                        self.hf_api.edit_discussion_comment(
                            repo_id=repo_id,
                            discussion_num=pr_num,
                            comment_id=first_comment_id,
                            new_content=migration_description,
                            repo_type="model"
                        )
                        
                        self.logger.info(f"Updated PR description for {repo_id}")
                    else:
                        self.logger.warning(f"Could not find first comment in PR {pr_num} for {repo_id}")
                    
                except Exception as e:
                    if self.verbose:
                        import traceback
                        self.logger.warning(f"Failed to update PR description for {repo_id}: {e}")
                        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                    else:
                        self.logger.warning(f"Failed to update PR description for {repo_id}: {e}")
                    # Continue anyway - PR exists even if description update failed
            
            return pr_url
            
        except Exception as e:
            if self.verbose:
                import traceback
                self.logger.error(f"Failed to upload changes for {repo_id}: {e}")
                self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            else:
                self.logger.error(f"Failed to upload changes for {repo_id}: {e}")
            return None  # Return None instead of raising to allow retry
    
    def _preview_pr_content(self, repo_id: str, title: str, description: str, files_modified: list):
        """Preview the pull request content before creation"""
        print("\n" + "="*80)
        print(f"📋 PULL REQUEST PREVIEW for {repo_id}")
        print("="*80)
        
        print(f"\n📝 TITLE:")
        print(f"{title}")
        
        print(f"\n📄 DESCRIPTION:")
        print("-" * 60)
        # Add proper indentation for the description
        for line in description.split('\n'):
            print(line)
        print("-" * 60)
        
        print(f"\n📁 FILES TO BE MODIFIED ({len(files_modified)} files):")
        for file_path in sorted(files_modified):
            print(f"  • {file_path}")
        
        print("\n" + "="*80)

    def cleanup_temp_directory(self, temp_path: str):
        """Clean up temporary working directory"""
        if not temp_path or not os.path.exists(temp_path):
            return
            
        try:
            import shutil
            shutil.rmtree(temp_path)
            self.logger.info(f"Cleaned up temporary directory: {temp_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary directory {temp_path}: {e}")


