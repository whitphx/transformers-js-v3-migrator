import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class RepoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RepoProgress:
    repo_id: str
    status: RepoStatus
    timestamp: str
    error_message: Optional[str] = None
    pr_url: Optional[str] = None


@dataclass
class SessionConfig:
    limit: int
    org_filter: Optional[str] = None
    repo_name_filter: Optional[str] = None
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
        params_str = f"{config.org_filter or 'all'}_{config.repo_name_filter or 'all'}_{','.join(sorted(config.exclude_orgs))}"
        
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
                    "pending": 0
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

    def filter_unprocessed_repos(self, repos: List[str]) -> List[str]:
        """Filter out repositories that have been processed in any previous session"""
        global_processed = self.get_global_processed_repos()
        unprocessed = [repo for repo in repos if repo not in global_processed]
        
        skipped_count = len(repos) - len(unprocessed)
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} previously processed repositories")
        
        return unprocessed

    def update_repo_status(self, session_id: str, repo_id: str, status: RepoStatus, 
                          error_message: Optional[str] = None, pr_url: Optional[str] = None):
        """Update the status of a repository in the session"""
        session_data = self.load_session(session_id)
        
        # Ensure repos dict exists
        if "repos" not in session_data:
            session_data["repos"] = {}
        
        progress = RepoProgress(
            repo_id=repo_id,
            status=status,
            timestamp=datetime.now().isoformat(),
            error_message=error_message,
            pr_url=pr_url
        )
        
        # Convert enum to string for JSON serialization
        progress_dict = asdict(progress)
        progress_dict["status"] = status.value
        
        session_data["repos"][repo_id] = progress_dict
        
        # Update stats
        self._update_session_stats(session_data)
        
        # If completed successfully, add to global processed list
        # Note: Even dry_run completions are added to global processed to avoid reprocessing
        if status == RepoStatus.COMPLETED:
            self.add_to_global_processed(repo_id)
        
        self.save_session(session_id, session_data)

    def _update_session_stats(self, session_data: Dict):
        """Update session statistics"""
        stats = {
            "total": len(session_data["repos"]),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "pending": 0
        }
        
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