import os
import re
from typing import Optional
from ..migration_types import BaseMigration, MigrationType, MigrationResult, MigrationStatus
from ..lib.ai_readme_migrator import AIReadmeMigrator


class ReadmeSamplesMigration(BaseMigration):
    """Migration for updating README.md sample code using AI"""
    
    def __init__(self):
        super().__init__()
        self.ai_migrator = None
    
    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.README_SAMPLES
    
    @property
    def description(self) -> str:
        return "Update README.md sample code from Transformers.js v2 to v3 using AI"
    
    def is_applicable(self, repo_path: str, repo_id: str) -> bool:
        """Check if README.md exists and contains v2 code"""
        readme_path = os.path.join(repo_path, "README.md")
        
        if not os.path.exists(readme_path):
            return False
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._contains_transformersjs_v2_code(content)
        except Exception as e:
            self.logger.error(f"Error checking README for {repo_id}: {e}")
            return False
    
    def apply_migration(self, repo_path: str, repo_id: str, interactive: bool = True) -> MigrationResult:
        """Apply README.md migration using AI"""
        readme_path = os.path.join(repo_path, "README.md")
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Initialize AI migrator if not already done
            if self.ai_migrator is None:
                self.ai_migrator = AIReadmeMigrator()
            
            # Use AI service for migration
            updated_content = self.ai_migrator.migrate_readme_content(original_content, repo_id, interactive=interactive)
            
            if updated_content and updated_content != original_content:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                self.logger.info(f"Successfully updated README.md for {repo_id}")
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.COMPLETED,
                    changes_made=True,
                    files_modified=["README.md"]
                )
            else:
                self.logger.info(f"No changes needed for README.md in {repo_id}")
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.SKIPPED,
                    changes_made=False
                )
        
        except Exception as e:
            error_msg = f"Error migrating README for {repo_id}: {e}"
            self.logger.error(error_msg)
            return MigrationResult(
                migration_type=self.migration_type,
                status=MigrationStatus.FAILED,
                changes_made=False,
                error_message=error_msg
            )
    
    def _contains_transformersjs_v2_code(self, content: str) -> bool:
        """Check if content contains Transformers.js v2 specific code"""
        v2_patterns = [
            r'@xenova/transformers',
            r'from\s+[\'"]@xenova/transformers[\'"]',
            r'import.*@xenova/transformers',
            r'cdn\.jsdelivr\.net/npm/@xenova/transformers'
        ]
        
        for pattern in v2_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
