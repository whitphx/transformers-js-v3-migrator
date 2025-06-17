import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from .migration_types import MigrationType, MigrationStatus


class RepoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MigrationProgress:
    migration_type: str  # MigrationType.value
    status: str  # MigrationStatus.value
    timestamp: str
    error_message: Optional[str] = None
    pr_url: Optional[str] = None
    files_modified: List[str] = None
    
    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []


@dataclass
class RepoProgress:
    repo_id: str
    status: RepoStatus
    timestamp: str
    error_message: Optional[str] = None
    migrations: Dict[str, MigrationProgress] = None  # migration_type -> progress
    
    def __post_init__(self):
        if self.migrations is None:
            self.migrations = {}


@dataclass
class SessionConfig:
    limit: int
    exact_repo: Optional[str] = None
    repo_search: Optional[str] = None
    author_filter: Optional[str] = None
    exclude_orgs: List[str] = None
    dry_run: bool = False

    def __post_init__(self):
        if self.exclude_orgs is None:
            self.exclude_orgs = []


class SessionManager:
    def __init__(self, sessions_dir: str = ".sessions"):
        self.sessions_dir = sessions_dir
        self.global_processed_file = os.path.join(sessions_dir, "global_processed.json")
        self.logger = logging.getLogger(__name__)
        
        # Create sessions directory if it doesn't exist
        os.makedirs(sessions_dir, exist_ok=True)

    def generate_session_id(self, config: SessionConfig) -> str:
        """Generate a unique session ID based on search parameters"""
        # Create a string representation of the search parameters
        params_str = f"{config.exact_repo or 'none'}_{config.repo_search or 'none'}_{config.author_filter or 'none'}_{','.join(sorted(config.exclude_orgs))}"
        
        # Generate a hash for the session ID
        session_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"session_{session_hash}"

    def get_session_file(self, session_id: str) -> str:
        """Get the file path for a session"""
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    def load_session(self, session_id: str) -> Dict:
        """Load session data from file"""
        session_file = self.get_session_file(session_id)
        
        if not os.path.exists(session_file):
            return {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "config": None,
                "repos": {},
                "stats": {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "pending": 0,
                    "migrations": {}  # migration_type -> {completed: 0, failed: 0, pending: 0}
                }
            }
        
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading session {session_id}: {e}")
            return {}

    def save_session(self, session_id: str, session_data: Dict):
        """Save session data to file"""
        session_file = self.get_session_file(session_id)
        session_data["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving session {session_id}: {e}")

    def get_global_processed_repos(self) -> Set[str]:
        """Get set of all repositories that have been processed across all sessions"""
        if not os.path.exists(self.global_processed_file):
            return set()
        
        try:
            with open(self.global_processed_file, 'r') as f:
                data = json.load(f)
                return set(data.get("processed_repos", []))
        except Exception as e:
            self.logger.error(f"Error loading global processed repos: {e}")
            return set()

    def add_to_global_processed(self, repo_id: str):
        """Add a repository to the global processed list"""
        processed_repos = self.get_global_processed_repos()
        processed_repos.add(repo_id)
        
        try:
            with open(self.global_processed_file, 'w') as f:
                json.dump({
                    "processed_repos": list(processed_repos),
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error updating global processed repos: {e}")

    def initialize_session(self, config: SessionConfig) -> tuple[str, Dict]:
        """Initialize or resume a session"""
        session_id = self.generate_session_id(config)
        session_data = self.load_session(session_id)
        
        # Update config if this is a new session or config has changed
        config_dict = asdict(config)
        if session_data.get("config") != config_dict:
            session_data["config"] = config_dict
            self.save_session(session_id, session_data)
            self.logger.info(f"Updated config for session {session_id}")
        
        self.logger.info(f"Initialized session: {session_id}")
        return session_id, session_data

    def filter_unprocessed_repos(self, repos: List[str], current_mode: str = "normal") -> List[str]:
        """Filter out repositories that have been fully processed in any previous session
        
        A repository is considered fully processed only if ALL its migrations are in
        COMPLETED, SKIPPED, or NOT_APPLICABLE status. Repositories with only DRY_RUN
        or LOCAL migrations should be available for re-processing in other modes.
        """
        unprocessed = []
        skipped_count = 0
        
        for repo in repos:
            if self._is_repo_fully_processed(repo, current_mode):
                skipped_count += 1
                self.logger.debug(f"Skipping {repo} - fully processed")
            else:
                unprocessed.append(repo)
        
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} previously processed repositories")
        
        return unprocessed
    
    def _is_repo_fully_processed(self, repo_id: str, current_mode: str) -> bool:
        """Check if a repository is fully processed and should be skipped"""
        # First check if it's in the global processed list
        global_processed = self.get_global_processed_repos()
        if repo_id not in global_processed:
            return False
        
        # If it's in global processed, we need to check the actual migration statuses
        # across all sessions to see if it's truly complete
        return self._check_repo_completion_across_sessions(repo_id, current_mode)
    
    def _check_repo_completion_across_sessions(self, repo_id: str, current_mode: str) -> bool:
        """Check if a repo is fully completed across all sessions"""
        import glob
        
        # Get all session files
        session_files = glob.glob(os.path.join(self.sessions_dir, "session_*.json"))
        
        # Collect all migration statuses for this repo across all sessions
        all_migration_statuses = {}
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    
                repo_data = session_data.get("repos", {}).get(repo_id, {})
                migrations = repo_data.get("migrations", {})
                
                # Update our view of migration statuses (later sessions override earlier ones)
                for migration_type, migration_data in migrations.items():
                    all_migration_statuses[migration_type] = migration_data.get("status")
            except Exception as e:
                self.logger.warning(f"Error reading session file {session_file}: {e}")
                continue
        
        if not all_migration_statuses:
            return False
        
        # Check if attempted migrations are truly complete
        # Note: NOT_APPLICABLE migrations are not saved, so missing types are OK
        for migration_type, status in all_migration_statuses.items():
            if status in [MigrationStatus.COMPLETED.value, MigrationStatus.SKIPPED.value]:
                continue
            elif status in [MigrationStatus.DRY_RUN.value, MigrationStatus.LOCAL.value]:
                # These statuses mean the repo can be re-processed in normal mode
                if current_mode == "normal":
                    return False
                # For dry_run and local modes, if there's already a DRY_RUN/LOCAL, we can skip
                elif current_mode in ["dry_run", "local"] and status == MigrationStatus.DRY_RUN.value:
                    continue
                elif current_mode == "local" and status == MigrationStatus.LOCAL.value:
                    continue
                else:
                    return False
            else:
                # PENDING, IN_PROGRESS, FAILED - not complete
                return False
        
        # Repository is complete if all attempted migrations are in final states
        # Missing migration types (NOT_APPLICABLE) don't block completion
        return len(all_migration_statuses) > 0

    def update_repo_status(self, session_id: str, repo_id: str, status: RepoStatus, 
                          error_message: Optional[str] = None):
        """Update the overall status of a repository in the session"""
        session_data = self.load_session(session_id)
        
        # Ensure repos dict exists
        if "repos" not in session_data:
            session_data["repos"] = {}
        
        # Get existing repo progress or create new one
        existing_data = session_data["repos"].get(repo_id, {})
        
        progress = RepoProgress(
            repo_id=repo_id,
            status=status,
            timestamp=datetime.now().isoformat(),
            error_message=error_message,
            migrations=existing_data.get("migrations", {})
        )
        
        # Convert enum to string for JSON serialization
        progress_dict = asdict(progress)
        progress_dict["status"] = status.value
        
        session_data["repos"][repo_id] = progress_dict
        
        # Update stats
        self._update_session_stats(session_data)
        
        # Only add to global processed list if status is COMPLETED
        # This ensures repos are only marked as globally processed when all migrations succeed
        if status == RepoStatus.COMPLETED:
            self.add_to_global_processed(repo_id)
        
        self.save_session(session_id, session_data)

    def update_migration_status(self, session_id: str, repo_id: str, migration_type: MigrationType,
                               status: MigrationStatus, error_message: Optional[str] = None,
                               pr_url: Optional[str] = None, files_modified: Optional[List[str]] = None):
        """Update the status of a specific migration for a repository"""
        session_data = self.load_session(session_id)
        
        # Ensure repos dict exists
        if "repos" not in session_data:
            session_data["repos"] = {}
        
        # Get existing repo data or create new one
        if repo_id not in session_data["repos"]:
            session_data["repos"][repo_id] = {
                "repo_id": repo_id,
                "status": RepoStatus.PENDING.value,
                "timestamp": datetime.now().isoformat(),
                "error_message": None,
                "migrations": {}
            }
        
        # Update migration progress
        migration_progress = MigrationProgress(
            migration_type=migration_type.value,
            status=status.value,
            timestamp=datetime.now().isoformat(),
            error_message=error_message,
            pr_url=pr_url,
            files_modified=files_modified or []
        )
        
        session_data["repos"][repo_id]["migrations"][migration_type.value] = asdict(migration_progress)
        
        # Update overall repo status based on migration statuses
        old_status = session_data["repos"][repo_id]["status"]
        self._update_repo_status_from_migrations(session_data, repo_id)
        new_status = session_data["repos"][repo_id]["status"]
        
        # Only add to global processed list when repo status changes to completed
        # This ensures upload success is required before marking as globally processed
        if old_status != RepoStatus.COMPLETED.value and new_status == RepoStatus.COMPLETED.value:
            self.add_to_global_processed(repo_id)
        
        # Update stats
        self._update_session_stats(session_data)
        
        self.save_session(session_id, session_data)

    def _update_repo_status_from_migrations(self, session_data: Dict, repo_id: str):
        """Update overall repo status based on individual migration statuses
        
        Note: Only migrations that were actually attempted/applicable are tracked.
        NOT_APPLICABLE migrations are not saved to allow future re-evaluation.
        """
        repo_data = session_data["repos"][repo_id]
        migrations = repo_data.get("migrations", {})
        
        if not migrations:
            return  # No migrations attempted yet, keep current status
        
        migration_statuses = [m["status"] for m in migrations.values()]
        
        # If any migration failed, repo should remain PENDING for retry (not FAILED)
        # This allows the repo to be retried in future sessions
        if any(status == MigrationStatus.FAILED.value for status in migration_statuses):
            repo_data["status"] = RepoStatus.PENDING.value  # Changed from FAILED to PENDING
        # If all attempted migrations are completed or skipped, mark repo as completed
        # Note: DRY_RUN and LOCAL status should not count as completion
        # Note: NOT_APPLICABLE migrations are not saved, so they don't affect completion
        elif all(status in [MigrationStatus.COMPLETED.value, MigrationStatus.SKIPPED.value] 
                for status in migration_statuses):
            repo_data["status"] = RepoStatus.COMPLETED.value
        # If any migration is in dry run or local state, keep repo as pending
        elif any(status in [MigrationStatus.DRY_RUN.value, MigrationStatus.LOCAL.value] for status in migration_statuses):
            repo_data["status"] = RepoStatus.PENDING.value
        # If any migration is in progress, mark repo as in progress
        elif any(status == MigrationStatus.IN_PROGRESS.value for status in migration_statuses):
            repo_data["status"] = RepoStatus.IN_PROGRESS.value
        else:
            repo_data["status"] = RepoStatus.PENDING.value

    def _update_session_stats(self, session_data: Dict):
        """Update session statistics"""
        stats = {
            "total": len(session_data["repos"]),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "pending": 0,
            "migrations": {}
        }
        
        # Count repo statuses
        for repo_data in session_data["repos"].values():
            status = repo_data["status"]
            if status == RepoStatus.COMPLETED.value:
                stats["completed"] += 1
            elif status == RepoStatus.FAILED.value:
                stats["failed"] += 1
            elif status == RepoStatus.SKIPPED.value:
                stats["skipped"] += 1
            else:
                stats["pending"] += 1
            
            # Count migration statuses
            for migration_type, migration_data in repo_data.get("migrations", {}).items():
                if migration_type not in stats["migrations"]:
                    stats["migrations"][migration_type] = {
                        "completed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "pending": 0,
                        "not_applicable": 0,
                        "dry_run": 0,
                        "local": 0
                    }
                
                migration_status = migration_data["status"]
                if migration_status == MigrationStatus.COMPLETED.value:
                    stats["migrations"][migration_type]["completed"] += 1
                elif migration_status == MigrationStatus.FAILED.value:
                    stats["migrations"][migration_type]["failed"] += 1
                elif migration_status == MigrationStatus.SKIPPED.value:
                    stats["migrations"][migration_type]["skipped"] += 1
                elif migration_status == MigrationStatus.NOT_APPLICABLE.value:
                    stats["migrations"][migration_type]["not_applicable"] += 1
                elif migration_status == MigrationStatus.DRY_RUN.value:
                    stats["migrations"][migration_type]["dry_run"] += 1
                elif migration_status == MigrationStatus.LOCAL.value:
                    stats["migrations"][migration_type]["local"] += 1
                else:
                    stats["migrations"][migration_type]["pending"] += 1
        
        session_data["stats"] = stats

    def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of the session progress"""
        session_data = self.load_session(session_id)
        return {
            "session_id": session_id,
            "created_at": session_data.get("created_at"),
            "updated_at": session_data.get("updated_at"),
            "config": session_data.get("config"),
            "stats": session_data.get("stats", {}),
            "recent_failures": [
                {
                    "repo_id": repo_id,
                    "error": repo_data.get("error_message"),
                    "timestamp": repo_data.get("timestamp")
                }
                for repo_id, repo_data in session_data.get("repos", {}).items()
                if repo_data.get("status") == RepoStatus.FAILED.value
            ][-5:]  # Last 5 failures
        }

    def list_sessions(self) -> List[Dict]:
        """List all existing sessions"""
        sessions = []
        
        if not os.path.exists(self.sessions_dir):
            return sessions
        
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json') and filename.startswith('session_'):
                session_id = filename[:-5]  # Remove .json extension
                try:
                    summary = self.get_session_summary(session_id)
                    sessions.append(summary)
                except Exception as e:
                    self.logger.error(f"Error loading session {session_id}: {e}")
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)