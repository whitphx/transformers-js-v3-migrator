import os
import re
from typing import Optional
from ..migration_types import BaseMigration, MigrationType, MigrationResult, MigrationStatus
from ..utils.ai_readme_migrator import AIReadmeMigrator


class ReadmeSamplesMigration(BaseMigration):
    """Migration for updating README.md sample code using AI"""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.ai_migrator = None

    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.README_SAMPLES

    @property
    def description(self) -> str:
        return "Update README.md for Transformers.js v3 compatibility and add missing installation/usage examples"

    def is_applicable(self, repo_path: str, repo_id: str) -> bool:
        """Check if README.md exists - let the AI decide if migration is needed"""
        readme_path = os.path.join(repo_path, "README.md")
        return os.path.exists(readme_path)


    def apply_migration(self, repo_path: str, repo_id: str, interactive: bool = True) -> MigrationResult:
        """Apply README.md migration using AI"""
        readme_path = os.path.join(repo_path, "README.md")

        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Initialize AI migrator if not already done
            if self.ai_migrator is None:
                self.ai_migrator = AIReadmeMigrator(verbose=self.verbose)

            # Use AI service for migration
            updated_content = self.ai_migrator.migrate_readme_content(original_content, repo_id, interactive=interactive)

            # Handle special return value for user rejection with skip
            if updated_content == "REJECT_SKIP":
                self.logger.info(f"User rejected changes for README.md in {repo_id} - leaving undone for future attempts")
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.FAILED,  # Mark as failed so repo isn't added to global processed list
                    changes_made=False,
                    error_message="User declined changes (ns) - repository left undone for future attempts"
                )

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
