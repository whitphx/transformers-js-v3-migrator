from typing import List, Optional
import logging
from huggingface_hub import HfApi
from .hub_client import HubClient
from .git_operations import GitOperations
from .migration_rules import MigrationRules
from .session_manager import SessionManager, SessionConfig, RepoStatus


class TransformersJSMigrator:
    def __init__(self, token: Optional[str] = None, mode: str = "normal"):
        self.token = token
        self.mode = mode  # "normal", "dry_run", or "preview"
        self.hub_client = HubClient(token)
        self.git_ops = GitOperations()
        self.migration_rules = MigrationRules()
        self.session_manager = SessionManager()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log mode information
        if self.mode == "preview":
            self.logger.info("Running in PREVIEW mode - no changes, no tracking")
        elif self.mode == "dry_run":
            self.logger.info("Running in DRY RUN mode - no changes, but tracking progress")
        else:
            self.logger.info("Running in NORMAL mode - making actual changes")

    def run_migration(self, limit: int = 10, org_filter: Optional[str] = None, 
                     repo_name_filter: Optional[str] = None, exclude_orgs: Optional[List[str]] = None,
                     resume: bool = False):
        """Run the migration process for Transformers.js v2 to v3"""
        
        # Handle preview mode - no session tracking
        if self.mode == "preview":
            self.logger.info("Starting migration preview (no session tracking)")
            
            # Search for repositories using Transformers.js
            all_repos = self.hub_client.search_transformersjs_repos(
                limit=limit,
                org_filter=org_filter,
                repo_name_filter=repo_name_filter,
                exclude_orgs=exclude_orgs
            )
            
            self.logger.info(f"Found {len(all_repos)} repositories (preview mode)")
            
            for i, repo in enumerate(all_repos, 1):
                self.logger.info(f"[PREVIEW {i}/{len(all_repos)}] Would process: {repo}")
            
            self.logger.info("Preview completed - no changes made, no progress tracked")
            return
        
        # Create session configuration for normal and dry_run modes
        config = SessionConfig(
            limit=limit,
            org_filter=org_filter,
            repo_name_filter=repo_name_filter,
            exclude_orgs=exclude_orgs or [],
            dry_run=(self.mode == "dry_run")
        )
        
        # Initialize or resume session
        session_id, session_data = self.session_manager.initialize_session(config)
        
        self.logger.info(f"Starting migration process (session: {session_id}, mode: {self.mode})")
        
        if resume and session_data.get("repos"):
            self.logger.info(f"Resuming session with {len(session_data['repos'])} existing repositories")
        
        # Search for repositories using Transformers.js
        all_repos = self.hub_client.search_transformersjs_repos(
            limit=limit,
            org_filter=org_filter,
            repo_name_filter=repo_name_filter,
            exclude_orgs=exclude_orgs
        )
        
        # Filter out globally processed repositories
        repos_to_process = self.session_manager.filter_unprocessed_repos(all_repos)
        
        self.logger.info(f"Found {len(repos_to_process)} new repositories to migrate")
        
        # Initialize session with pending repositories
        for repo in repos_to_process:
            if repo not in session_data.get("repos", {}):
                self.session_manager.update_repo_status(session_id, repo, RepoStatus.PENDING)
        
        # Process repositories
        failed_count = 0
        completed_count = 0
        
        for repo in repos_to_process:
            try:
                self.logger.info(f"Processing repository {completed_count + failed_count + 1}/{len(repos_to_process)}: {repo}")
                
                # Update status to in_progress
                self.session_manager.update_repo_status(session_id, repo, RepoStatus.IN_PROGRESS)
                
                # Migrate repository
                success, pr_url, error_msg = self.migrate_repository(repo)
                
                if success:
                    self.session_manager.update_repo_status(
                        session_id, repo, RepoStatus.COMPLETED, pr_url=pr_url
                    )
                    completed_count += 1
                    self.logger.info(f"✓ Successfully migrated {repo}")
                else:
                    self.session_manager.update_repo_status(
                        session_id, repo, RepoStatus.FAILED, error_message=error_msg
                    )
                    failed_count += 1
                    self.logger.error(f"✗ Failed to migrate {repo}: {error_msg}")
                    
            except Exception as e:
                error_msg = str(e)
                self.session_manager.update_repo_status(
                    session_id, repo, RepoStatus.FAILED, error_message=error_msg
                )
                failed_count += 1
                self.logger.error(f"✗ Failed to migrate {repo}: {error_msg}")
        
        # Final summary
        self.logger.info(f"Migration completed! Session: {session_id}")
        self.logger.info(f"Results: {completed_count} completed, {failed_count} failed")
        
        if failed_count > 0:
            self.logger.info(f"Use 'python main.py status {session_id}' to see failed repositories")
            if self.mode == "dry_run":
                self.logger.info(f"Use 'python main.py migrate --resume' to retry failed repositories")
            else:
                self.logger.info(f"Use 'python main.py migrate --resume' to retry failed repositories")
    
    def migrate_repository(self, repo_id: str) -> tuple[bool, Optional[str], Optional[str]]:
        """Migrate a single repository
        
        Returns:
            tuple: (success, pr_url, error_message)
        """
        try:
            if self.mode == "dry_run":
                self.logger.info(f"[DRY RUN] Would migrate {repo_id}")
                return True, None, None
            elif self.mode == "preview":
                # This shouldn't be called in preview mode, but just in case
                self.logger.info(f"[PREVIEW] Would migrate {repo_id}")
                return True, None, None
            
            # Clone repository
            repo_path = self.git_ops.clone_repo(repo_id)
            
            try:
                # Apply migration rules
                changes_made = self.migration_rules.apply_migrations(repo_path)
                
                if changes_made:
                    # Create PR with changes
                    pr_url = self.git_ops.create_migration_pr(repo_path, repo_id)
                    return True, pr_url, None
                else:
                    self.logger.info(f"No changes needed for {repo_id}")
                    return True, None, None
                    
            finally:
                # Cleanup
                self.git_ops.cleanup_repo(repo_path)
                
        except Exception as e:
            return False, None, str(e)