from typing import List, Optional
import logging
from huggingface_hub import HfApi
from .hub_client import HubClient
from .git_operations import GitOperations
from .migration_rules import MigrationRules
from .session_manager import SessionManager, SessionConfig, RepoStatus
from .migration_types import migration_registry, MigrationStatus
# Import migrations to register them
import src.migrations


class TransformersJSMigrator:
    def __init__(self, token: Optional[str] = None, mode: str = "normal"):
        self.token = token
        self.mode = mode  # "normal", "dry_run", or "preview"
        self.hub_client = HubClient(token)
        self.git_ops = GitOperations(token)
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
                     resume: bool = False, interactive: bool = True):
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
                success, pr_url, error_msg = self.migrate_repository(repo, session_id, interactive)
                
                if success:
                    self.session_manager.update_repo_status(
                        session_id, repo, RepoStatus.COMPLETED
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
    
    def migrate_repository(self, repo_id: str, session_id: str, interactive: bool = True) -> tuple[bool, Optional[str], Optional[str]]:
        """Migrate a single repository
        
        Returns:
            tuple: (success, pr_url, error_message)
        """
        try:
            if self.mode == "preview":
                # This shouldn't be called in preview mode, but just in case
                self.logger.info(f"[PREVIEW] Would migrate {repo_id}")
                return True, None, None
            
            # Download repository
            repo_path = self.git_ops.download_repo(repo_id)
            
            try:
                # Get applicable migrations for this repository
                applicable_migrations = migration_registry.get_applicable_migrations(repo_path, repo_id)
                
                if not applicable_migrations:
                    self.logger.info(f"No applicable migrations for {repo_id}")
                    return True, None, None
                
                overall_success = True
                pr_urls = []
                errors = []
                
                # Use the main session_id from the run_migration method
                # Process each migration type separately  
                for migration in applicable_migrations:
                    
                    try:
                        # Update migration status to in_progress
                        self.session_manager.update_migration_status(
                            session_id, repo_id, migration.migration_type, MigrationStatus.IN_PROGRESS
                        )
                        
                        if self.mode == "dry_run":
                            self.logger.info(f"[DRY RUN] Would apply {migration.migration_type.value} migration for {repo_id}")
                            # Mock successful migration for dry run
                            self.session_manager.update_migration_status(
                                session_id, repo_id, migration.migration_type, MigrationStatus.COMPLETED
                            )
                            continue
                        
                        # Apply the migration
                        result = migration.apply_migration(repo_path, repo_id, interactive)
                        
                        if result.changes_made and self.mode == "normal":
                            # Upload changes for this specific migration using HF Hub
                            if self.git_ops.upload_changes(
                                repo_path, repo_id, 
                                migration.migration_type.value,
                                migration.get_pr_title(),
                                migration.get_pr_description(),
                                result.files_modified
                            ):
                                # Create pull request for this migration
                                pr_url = self.git_ops.create_pull_request(
                                    repo_id, 
                                    migration.get_pr_title(),
                                    migration.get_pr_description(),
                                    migration.migration_type.value
                                )
                                pr_urls.append(pr_url)
                                
                                # Update migration status with PR URL
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, 
                                    MigrationStatus.COMPLETED, pr_url=pr_url,
                                    files_modified=result.files_modified
                                )
                                
                                self.logger.info(f"✓ {migration.migration_type.value} migration completed for {repo_id}")
                            else:
                                error_msg = f"Failed to upload {migration.migration_type.value} changes"
                                errors.append(error_msg)
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, 
                                    MigrationStatus.FAILED, error_message=error_msg
                                )
                                overall_success = False
                        else:
                            # No changes or dry run mode
                            status = MigrationStatus.COMPLETED if result.changes_made else MigrationStatus.SKIPPED
                            self.session_manager.update_migration_status(
                                session_id, repo_id, migration.migration_type, status,
                                files_modified=result.files_modified
                            )
                            
                    except Exception as e:
                        error_msg = f"Error in {migration.migration_type.value} migration: {str(e)}"
                        errors.append(error_msg)
                        self.session_manager.update_migration_status(
                            session_id, repo_id, migration.migration_type, 
                            MigrationStatus.FAILED, error_message=error_msg
                        )
                        overall_success = False
                        self.logger.error(error_msg)
            
                # Return overall result
                if overall_success:
                    return True, pr_urls[0] if pr_urls else None, None
                else:
                    return False, None, "; ".join(errors)
                    
            finally:
                # Clean up temporary working directory
                self.git_ops.cleanup_temp_directory(repo_path)
                
        except Exception as e:
            return False, None, str(e)