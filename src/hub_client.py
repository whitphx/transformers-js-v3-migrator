from typing import List, Optional
from huggingface_hub import HfApi, list_models
import logging


class HubClient:
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)
        self.logger = logging.getLogger(__name__)

    def search_transformersjs_repos(self, limit: int = 10, exact_repo: Optional[str] = None,
                                   repo_search: Optional[str] = None, author_filter: Optional[str] = None,
                                   exclude_orgs: Optional[List[str]] = None) -> List[str]:
        """Search for model repositories that use Transformers.js library with filtering options
        
        Args:
            limit: Maximum number of repositories to return
            exact_repo: Exact repository name (format: org/repo-name) - returns only this repo if it exists
            repo_search: Search repositories by name pattern
            author_filter: Filter repositories by author/organization name
            exclude_orgs: Exclude repositories from these organizations
        """
        try:
            # Handle exact repository match
            if exact_repo:
                self.logger.info(f"Looking for exact repository: {exact_repo}")
                try:
                    # Check if the exact repository exists and has transformers.js library
                    repo_info = self.api.repo_info(exact_repo, repo_type="model")
                    if repo_info and repo_info.library_name == "transformers.js":
                        self.logger.info(f"Found exact repository: {exact_repo}")
                        return [exact_repo]
                    else:
                        self.logger.warning(f"Repository {exact_repo} not found or doesn't use transformers.js")
                        return []
                except Exception as e:
                    self.logger.error(f"Error checking exact repository {exact_repo}: {e}")
                    return []
            
            # Build search parameters for general search
            search_params = {
                "library": "transformers.js",
                "limit": limit,
                "sort": "downloads",
                "direction": -1
            }
            
            # Add author filter if specified
            if author_filter:
                search_params["author"] = author_filter
                self.logger.info(f"Filtering by author: {author_filter}")
            
            # Add repository name search if specified
            if repo_search:
                search_params["search"] = repo_search
                self.logger.info(f"Searching repositories by name: {repo_search}")
            
            models = list_models(**search_params)
            repo_ids = [model.id for model in models]
            
            # Apply exclude_orgs filter (post-processing since list_models doesn't support exclusion)
            if exclude_orgs:
                original_count = len(repo_ids)
                repo_ids = [
                    repo_id for repo_id in repo_ids 
                    if not any(repo_id.startswith(f"{excluded_org}/") for excluded_org in exclude_orgs)
                ]
                self.logger.info(f"Excluded {original_count - len(repo_ids)} repositories from organizations: {exclude_orgs}")
            
            self.logger.info(f"Found {len(repo_ids)} Transformers.js repositories")
            return repo_ids
            
        except Exception as e:
            self.logger.error(f"Error searching repositories: {e}")
            return []

    def get_repo_info(self, repo_id: str) -> dict:
        """Get detailed information about a repository"""
        try:
            return self.api.repo_info(repo_id, repo_type="model")
        except Exception as e:
            self.logger.error(f"Error getting repo info for {repo_id}: {e}")
            return {}

    def check_transformersjs_version(self, repo_id: str) -> Optional[str]:
        """Check which version of Transformers.js the repository is using"""
        try:
            # Look for package.json or other version indicators
            files = self.api.list_repo_files(repo_id, repo_type="model")
            
            if "package.json" in files:
                # TODO: Read package.json to check version
                return "unknown"
            
            # Check for other version indicators in README or model files
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error checking version for {repo_id}: {e}")
            return None