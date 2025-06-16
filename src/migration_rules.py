import os
import json
import re
from typing import List, Dict, Any, Optional
import logging


class MigrationRules:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def apply_migrations(self, repo_path: str, repo_id: str) -> bool:
        """Apply all migration rules to the repository"""
        changes_made = False
        
        # Focus on README.md migration for now
        readme_path = os.path.join(repo_path, "README.md")
        if os.path.exists(readme_path):
            if self.migrate_readme(readme_path, repo_id):
                changes_made = True
                self.logger.info(f"Updated README.md for {repo_id}")
        else:
            self.logger.info(f"No README.md found for {repo_id}")
        
        return changes_made

    def migrate_config_json(self, config_path: str) -> bool:
        """Migrate config.json for Transformers.js v3 compatibility"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            original_config = config.copy()
            
            # Add transformers_version if not present
            if 'transformers_version' not in config:
                config['transformers_version'] = '4.0.0'
            
            # Update any v2-specific configurations
            # TODO: Add specific migration rules based on v2 to v3 changes
            
            if config != original_config:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.logger.info(f"Updated config.json at {config_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error migrating config.json at {config_path}: {e}")
        
        return False

    def migrate_readme(self, readme_path: str, repo_id: str) -> bool:
        """Migrate README.md for v3 compatibility using AI service (mocked for now)"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Check if README contains Transformers.js v2 code
            if not self._contains_transformersjs_v2_code(content):
                self.logger.info(f"README.md for {repo_id} doesn't contain v2 code, skipping")
                return False
            
            self.logger.info(f"Found Transformers.js v2 code in README.md for {repo_id}")
            
            # Mock AI service call for now
            updated_content = self._mock_ai_migration(content, repo_id)
            
            if updated_content and updated_content != original_content:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                self.logger.info(f"Updated README.md for {repo_id} using AI migration")
                return True
            else:
                self.logger.info(f"No changes needed for README.md in {repo_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error migrating README for {repo_id}: {e}")
            return False

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

    def _mock_ai_migration(self, content: str, repo_id: str) -> Optional[str]:
        """Mock AI service call to migrate README content from v2 to v3
        
        TODO: Replace with actual AI service integration
        """
        self.logger.info(f"[MOCK] Calling AI service to migrate README for {repo_id}")
        
        # Simple regex-based migration for now
        updated_content = content
        
        # Update package imports
        updated_content = re.sub(
            r'@xenova/transformers',
            '@huggingface/transformers',
            updated_content
        )
        
        # Update CDN links
        updated_content = re.sub(
            r'https://cdn\.jsdelivr\.net/npm/@xenova/transformers(@[\d\.]+)?',
            'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3',
            updated_content
        )
        
        # Update import statements
        updated_content = re.sub(
            r'from\s+[\'"]@xenova/transformers[\'"]',
            'from "@huggingface/transformers"',
            updated_content
        )
        
        # Add migration notice
        if updated_content != content:
            migration_notice = "\n<!-- This README has been automatically migrated to Transformers.js v3 -->\n"
            updated_content = migration_notice + updated_content
        
        self.logger.info(f"[MOCK] AI migration completed for {repo_id}")
        return updated_content

    def migrate_code_examples(self, file_path: str) -> bool:
        """Migrate code examples in JS/HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            content = self.update_code_examples_in_text(content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Updated code examples in {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error migrating code examples in {file_path}: {e}")
        
        return False

    def update_code_examples_in_text(self, content: str) -> str:
        """Update code examples within text content"""
        # Update package imports
        content = re.sub(
            r'@xenova/transformers',
            '@huggingface/transformers',
            content
        )
        
        # Update specific API changes (placeholder for actual v2->v3 changes)
        # TODO: Add specific API migration rules
        
        return content