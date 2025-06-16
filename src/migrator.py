from typing import List, Optional
import logging
from huggingface_hub import HfApi
from .hub_client import HubClient
from .git_operations import GitOperations
from .migration_rules import MigrationRules


class TransformersJSMigrator:
    def __init__(self, token: Optional[str] = None, dry_run: bool = False):
        self.token = token
        self.dry_run = dry_run
        self.hub_client = HubClient(token)
        self.git_ops = GitOperations()
        self.migration_rules = MigrationRules()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run_migration(self, limit: int = 10, org_filter: Optional[str] = None, 
                     repo_name_filter: Optional[str] = None, exclude_orgs: Optional[List[str]] = None):
        """Run the migration process for Transformers.js v2 to v3"""
        filters = {
            'org_filter': org_filter,
            'repo_name_filter': repo_name_filter,
            'exclude_orgs': exclude_orgs or []
        }
        
        self.logger.info(f"Starting migration process (limit: {limit}, dry_run: {self.dry_run})")
        if any(filters.values()):
            self.logger.info(f"Applied filters: {filters}")
        
        # Search for repositories using Transformers.js
        repos = self.hub_client.search_transformersjs_repos(limit=limit, **filters)
        self.logger.info(f"Found {len(repos)} repositories to migrate")
        
        for repo in repos:
            try:
                self.migrate_repository(repo)
            except Exception as e:
                self.logger.error(f"Failed to migrate {repo}: {e}")
    
    def migrate_repository(self, repo_id: str):
        """Migrate a single repository"""
        self.logger.info(f"Migrating repository: {repo_id}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would migrate {repo_id}")
            return
        
        # Clone repository
        repo_path = self.git_ops.clone_repo(repo_id)
        
        try:
            # Apply migration rules
            changes_made = self.migration_rules.apply_migrations(repo_path)
            
            if changes_made:
                # Create PR with changes
                self.git_ops.create_migration_pr(repo_path, repo_id)
                self.logger.info(f"Created migration PR for {repo_id}")
            else:
                self.logger.info(f"No changes needed for {repo_id}")
                
        finally:
            # Cleanup
            self.git_ops.cleanup_repo(repo_path)