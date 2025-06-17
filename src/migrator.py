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
    def __init__(self, token: Optional[str] = None, mode: str = "normal", verbose: bool = False):
        self.token = token
        self.mode = mode  # "normal", "dry_run", "preview", or "local"
        self.verbose = verbose
        self.hub_client = HubClient(token)
        self.git_ops = GitOperations(token, verbose=verbose)
        self.migration_rules = MigrationRules()
        self.session_manager = SessionManager()
        
        # Set logging level based on verbose mode
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        # Log mode information
        if self.verbose:
            self.logger.info("Running in VERBOSE mode - detailed error information will be shown")
        if self.mode == "preview":
            self.logger.info("Running in PREVIEW mode - no changes, no tracking")
        elif self.mode == "dry_run":
            self.logger.info("Running in DRY RUN mode - no changes, but tracking progress")
        elif self.mode == "local":
            self.logger.info("Running in LOCAL mode - applying changes locally but no commits/pushes")
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
                
                # Note: Don't call update_repo_status here - let individual migration status updates
                # determine the overall repo status via _update_repo_status_from_migrations
                
                if success:
                    completed_count += 1
                    self.logger.info(f"✓ Successfully migrated {repo}")
                else:
                    failed_count += 1
                    self.logger.error(f"✗ Failed to migrate {repo}: {error_msg}")
                    
            except Exception as e:
                error_msg = str(e)
                if self.verbose:
                    import traceback
                    self.logger.error(f"✗ Failed to migrate {repo}: {error_msg}")
                    self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                else:
                    self.logger.error(f"✗ Failed to migrate {repo}: {error_msg}")
                self.session_manager.update_repo_status(
                    session_id, repo, RepoStatus.FAILED, error_message=error_msg
                )
                failed_count += 1
        
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
                # Get all migrations and check applicability
                all_migrations = migration_registry.get_all_migrations()
                applicable_migrations = []
                
                # Load existing session data to check previous migration statuses
                session_data = self.session_manager.load_session(session_id)
                existing_repo_data = session_data.get("repos", {}).get(repo_id, {})
                existing_migrations = existing_repo_data.get("migrations", {})
                
                # Check each migration type and track its status
                for migration_class in all_migrations:
                    migration_type_name = migration_class.migration_type.value
                    
                    # Check if this migration was already completed in a previous run
                    existing_migration = existing_migrations.get(migration_type_name, {})
                    existing_status = existing_migration.get("status")
                    
                    # Skip migrations that are already completed (but allow re-running dry-run and local modes)
                    if existing_status in [MigrationStatus.COMPLETED.value, MigrationStatus.SKIPPED.value, MigrationStatus.NOT_APPLICABLE.value]:
                        self.logger.info(f"Skipping {migration_type_name} for {repo_id} - already {existing_status}")
                        continue
                    elif existing_status == MigrationStatus.DRY_RUN.value and self.mode != "dry_run":
                        # Previous run was dry-run, but now we're in normal/local mode - allow re-running
                        self.logger.info(f"Re-running {migration_type_name} for {repo_id} - previous was dry-run, now {self.mode}")
                    elif existing_status == MigrationStatus.LOCAL.value and self.mode == "normal":
                        # Previous run was local, but now we're in normal mode - allow re-running
                        self.logger.info(f"Re-running {migration_type_name} for {repo_id} - previous was local, now normal")
                    elif existing_status in [MigrationStatus.DRY_RUN.value, MigrationStatus.LOCAL.value]:
                        # Same mode as before, or going from normal to dry-run/local - skip
                        self.logger.info(f"Skipping {migration_type_name} for {repo_id} - already {existing_status}")
                        continue
                    
                    try:
                        # Create a new instance with verbose mode for this session
                        migration = migration_class.__class__(verbose=self.verbose)
                        
                        if migration.is_applicable(repo_path, repo_id):
                            applicable_migrations.append(migration)
                            self.logger.debug(f"Migration {migration.migration_type.value} is applicable to {repo_id}")
                        else:
                            # Mark non-applicable migrations as NOT_APPLICABLE
                            self.session_manager.update_migration_status(
                                session_id, repo_id, migration.migration_type, MigrationStatus.NOT_APPLICABLE
                            )
                            self.logger.debug(f"Migration {migration.migration_type.value} is not applicable to {repo_id}")
                    except Exception as e:
                        # Mark migrations with errors as FAILED
                        error_msg = f"Error checking applicability: {str(e)}"
                        self.session_manager.update_migration_status(
                            session_id, repo_id, migration_class.migration_type, MigrationStatus.FAILED, error_message=error_msg
                        )
                        if self.verbose:
                            import traceback
                            self.logger.error(f"Error checking applicability of {migration_class.migration_type.value} for {repo_id}: {e}")
                            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                        else:
                            self.logger.error(f"Error checking applicability of {migration_class.migration_type.value} for {repo_id}: {e}")
                
                if not applicable_migrations:
                    self.logger.info(f"No applicable migrations for {repo_id}")
                    # All migrations have been marked as NOT_APPLICABLE, so repo should be marked as completed
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
                            # Mark as dry run - this should not mark repo as completed
                            self.session_manager.update_migration_status(
                                session_id, repo_id, migration.migration_type, MigrationStatus.DRY_RUN
                            )
                            continue
                        
                        # Apply the migration
                        result = migration.apply_migration(repo_path, repo_id, interactive)
                        
                        if result.changes_made and self.mode == "local":
                            # Local mode: apply changes but don't upload
                            self.logger.info(f"[LOCAL] Applied {migration.migration_type.value} migration for {repo_id} (no upload)")
                            self.session_manager.update_migration_status(
                                session_id, repo_id, migration.migration_type, MigrationStatus.LOCAL,
                                files_modified=result.files_modified
                            )
                        elif result.changes_made and self.mode == "normal":
                            # Upload changes for this specific migration using HF Hub
                            pr_url = self.git_ops.upload_changes(
                                repo_path, repo_id, 
                                migration.migration_type.value,
                                migration.get_pr_title(),
                                migration.get_pr_description(),
                                result.files_modified,
                                interactive
                            )
                            
                            if pr_url:
                                pr_urls.append(pr_url)
                                
                                # Update migration status with PR URL
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, 
                                    MigrationStatus.COMPLETED, pr_url=pr_url,
                                    files_modified=result.files_modified
                                )
                                
                                self.logger.info(f"✓ {migration.migration_type.value} migration completed for {repo_id}")
                            else:
                                error_msg = f"Failed to upload {migration.migration_type.value} changes - leaving repo undone for retry"
                                errors.append(error_msg)
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, 
                                    MigrationStatus.FAILED, error_message=error_msg
                                )
                                overall_success = False
                        else:
                            # Check the actual migration result status
                            if result.status == MigrationStatus.FAILED:
                                # Migration failed
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, 
                                    MigrationStatus.FAILED, error_message=result.error_message
                                )
                                errors.append(f"{migration.migration_type.value} migration failed: {result.error_message}")
                                overall_success = False
                                self.logger.error(f"✗ {migration.migration_type.value} migration failed for {repo_id}: {result.error_message}")
                            elif result.changes_made and self.mode not in ["normal", "local"]:
                                # Migration completed with changes but not uploaded (dry run mode only)
                                # In normal/local mode, if we reach here it means upload wasn't attempted due to no changes_made
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, MigrationStatus.COMPLETED,
                                    files_modified=result.files_modified
                                )
                            else:
                                # No changes needed or migration was skipped
                                status = MigrationStatus.COMPLETED if result.status == MigrationStatus.COMPLETED else MigrationStatus.SKIPPED
                                self.session_manager.update_migration_status(
                                    session_id, repo_id, migration.migration_type, status,
                                    files_modified=result.files_modified
                                )
                            
                    except Exception as e:
                        error_msg = f"Error in {migration.migration_type.value} migration: {str(e)}"
                        errors.append(error_msg)
                        if self.verbose:
                            import traceback
                            self.logger.error(error_msg)
                            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                        else:
                            self.logger.error(error_msg)
                        self.session_manager.update_migration_status(
                            session_id, repo_id, migration.migration_type, 
                            MigrationStatus.FAILED, error_message=error_msg
                        )
                        overall_success = False
            
                # Return overall result
                if overall_success:
                    return True, pr_urls[0] if pr_urls else None, None
                else:
                    return False, None, "; ".join(errors)
                    
            finally:
                # Clean up temporary working directory
                self.git_ops.cleanup_temp_directory(repo_path)
                
        except Exception as e:
            if self.verbose:
                import traceback
                self.logger.error(f"Repository migration failed for {repo_id}: {e}")
                self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return False, None, str(e)