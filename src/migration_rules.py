import os
import json
import re
from typing import List, Dict, Any
import logging


class MigrationRules:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def apply_migrations(self, repo_path: str) -> bool:
        """Apply all migration rules to the repository"""
        changes_made = False
        
        # Check for files that need migration
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file == "config.json":
                    if self.migrate_config_json(file_path):
                        changes_made = True
                
                elif file == "README.md":
                    if self.migrate_readme(file_path):
                        changes_made = True
                
                elif file.endswith(('.js', '.html', '.md')):
                    if self.migrate_code_examples(file_path):
                        changes_made = True
        
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

    def migrate_readme(self, readme_path: str) -> bool:
        """Migrate README.md for v3 compatibility"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Update import statements
            content = re.sub(
                r'import\s+{\s*([^}]+)\s*}\s+from\s+[\'"]@xenova/transformers[\'"]',
                r'import { \1 } from "@huggingface/transformers"',
                content
            )
            
            # Update CDN links
            content = re.sub(
                r'https://cdn\.jsdelivr\.net/npm/@xenova/transformers@[\d\.]+',
                'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3',
                content
            )
            
            # Update code examples
            content = self.update_code_examples_in_text(content)
            
            if content != original_content:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Updated README.md at {readme_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error migrating README at {readme_path}: {e}")
        
        return False

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