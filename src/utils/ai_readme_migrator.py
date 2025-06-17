import os
import logging
from typing import Optional
from anthropic import Anthropic


class AIReadmeMigrator:
    """AI-powered README migrator using external LLM API for Transformers.js v2 to v3 migration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for AI README migration")
        
        self.client = Anthropic(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
    
    def migrate_readme_content(self, content: str, repo_id: str, interactive: bool = True) -> Optional[str]:
        """Migrate README content from Transformers.js v2 to v3 using AI"""
        
        try:
            self.logger.info(f"Calling AI service to migrate README for {repo_id}")
            
            # Create the migration prompt
            prompt = self._create_migration_prompt(content, repo_id)
            
            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            migrated_content = response.content[0].text.strip()
            
            # Validate the response
            if not self._validate_migration(content, migrated_content):
                self.logger.error(f"AI README migration validation failed for {repo_id}")
                return None
            
            # Interactive mode: show changes and ask for confirmation
            if interactive:
                user_choice = self._show_changes_and_confirm(content, migrated_content, repo_id)
                if user_choice == "accept":
                    pass  # Continue with migration
                elif user_choice == "reject_skip":
                    self.logger.info(f"User declined changes for {repo_id} - leaving undone for future attempts")
                    return "REJECT_SKIP"  # Special return value to indicate skip without marking done
                elif user_choice == "reject_done":
                    self.logger.info(f"User declined changes for {repo_id} - marking as done")
                    return None  # Standard rejection, will be marked as completed/skipped
                else:
                    # This shouldn't happen, but handle gracefully
                    self.logger.info(f"User declined changes for {repo_id}")
                    return None
            
            self.logger.info(f"AI README migration completed successfully for {repo_id}")
            return migrated_content
                
        except Exception as e:
            self.logger.error(f"Error during AI README migration for {repo_id}: {e}")
            return None
    
    def _create_migration_prompt(self, content: str, repo_id: str) -> str:
        """Create a detailed prompt for the AI README migration"""
        
        example_diff = """
Example migration (v2 -> v3):

BEFORE:
```js
// npm i @xenova/transformers
import { pipeline } from '@xenova/transformers';

let url = 'https://example.com/audio.wav';
let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
let output = await transcriber(url);
```

AFTER:
```js
import { pipeline } from '@huggingface/transformers';

// Create speech recognition pipeline
const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');

// Transcribe audio from URL
const url = 'https://example.com/audio.wav';
const output = await transcriber(url);
```
"""
        
        prompt = f"""You are migrating a Transformers.js model repository README from v2 to v3. Your task is to update the README content while preserving its original structure and purpose.

## CRITICAL REQUIREMENTS:
1. **Output only the migrated README content** - NO wrapper text, explanations, or meta-commentary
2. **Preserve original structure** - Keep the same sections, formatting, and overall organization
3. **Minimal changes only** - Only update what's necessary for v3 compatibility
4. **PRESERVE FRONTMATTER** - Keep all YAML frontmatter (content between --- lines) exactly as-is

## Required Changes:
1. **Package name**: Change `@xenova/transformers` to `@huggingface/transformers`
2. **Installation instructions**: ALWAYS add installation instructions when there are code examples, even if not explicitly missing
3. **Modern JavaScript**: Use `const` instead of `let` for variables that aren't reassigned
4. **Remove inline install comments**: Remove `// npm i @xenova/transformers` comments from code blocks
5. **Add semicolons**: Ensure statements end with semicolons where appropriate
6. **Update quantization syntax**: Change `{ quantized: false }` to `{ dtype: "fp32" }` and show available options

## Installation Section Template:
When adding installation instructions, use this format before the first code example:

```markdown
If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

You can then use the model as follows:
```

{example_diff}

## STRICT GUIDELINES:
- **NEVER remove frontmatter** - Keep all YAML metadata between --- lines exactly as-is
- DO NOT add explanatory text about what the code does
- DO NOT move example outputs or change code structure  
- DO NOT add sections that weren't in the original
- DO NOT add wrapper text like "Here is the migrated content"
- PRESERVE comments that are example outputs (like "// Found car at...")
- Keep the exact same markdown structure and sections
- Return ONLY the migrated README content, nothing else

## Repository: {repo_id}

## Original README Content:
{content}

## MIGRATED README (output only this):"""

        return prompt
    
    def _validate_migration(self, original: str, migrated: str) -> bool:
        """Validate the AI README migration result"""
        
        # Basic validation checks
        if not migrated or len(migrated) < 50:
            self.logger.error("README migration result is too short or empty")
            return False
        
        # Check for unwanted wrapper text
        unwanted_phrases = [
            "here is the migrated",
            "here's the migrated", 
            "migrated readme content:",
            "migrated content:",
            "```markdown\n---",  # Check if it starts with markdown wrapper
        ]
        migrated_lower = migrated.lower()
        for phrase in unwanted_phrases:
            if phrase in migrated_lower:
                self.logger.error(f"README migration contains unwanted wrapper text: '{phrase}'")
                return False
        
        # Check if the migration actually changed v2 to v3
        if '@xenova/transformers' in migrated:
            self.logger.error("README migration failed: Still contains @xenova/transformers")
            return False
        
        # Check if it contains the expected v3 package name (only if original had v2)
        if '@huggingface/transformers' not in migrated and '@xenova/transformers' in original:
            self.logger.error("README migration failed: Missing @huggingface/transformers")
            return False
        
        # Check if the content structure is preserved (reasonable length)
        length_ratio = len(migrated) / len(original)
        if length_ratio < 0.7 or length_ratio > 1.5:
            self.logger.error(f"README migration result length seems wrong: {length_ratio:.2f}x original")
            return False
        
        # Check that frontmatter is preserved if it existed in original
        original_has_frontmatter = original.strip().startswith('---')
        migrated_has_frontmatter = migrated.strip().startswith('---')
        
        if original_has_frontmatter and not migrated_has_frontmatter:
            self.logger.error("README migration removed frontmatter - this is not allowed")
            return False
        
        # Check that it starts properly (should start with frontmatter or content, not wrapper text)
        if not (migrated.startswith('---') or migrated.startswith('#') or migrated.startswith('<!') or migrated.strip().startswith('##')):
            first_line = migrated.split('\n')[0]
            if any(word in first_line.lower() for word in ['here', 'migrated', 'content']):
                self.logger.error(f"README migration starts with unwanted text: '{first_line}'")
                return False
        
        return True
    
    def _show_changes_and_confirm(self, original: str, migrated: str, repo_id: str) -> str:
        """Show the changes to the user and ask for confirmation"""
        import difflib
        
        print(f"\n{'='*80}")
        print(f"AI Migration Suggested Changes for: {repo_id}")
        print(f"{'='*80}")
        print(f"Commit message: 📝 Update README.md samples for Transformers.js v3")
        print(f"{'='*80}")
        
        # Generate and display diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile="README.md (original)",
            tofile="README.md (migrated)",
            lineterm=""
        )
        
        diff_output = list(diff)
        if diff_output:
            print("\nChanges:")
            for line in diff_output:
                line = line.rstrip()
                if line.startswith('---') or line.startswith('+++'):
                    print(f"\033[1m{line}\033[0m")  # Bold
                elif line.startswith('-'):
                    print(f"\033[31m{line}\033[0m")  # Red
                elif line.startswith('+'):
                    print(f"\033[32m{line}\033[0m")  # Green
                elif line.startswith('@@'):
                    print(f"\033[36m{line}\033[0m")  # Cyan
                else:
                    print(line)
        else:
            print("\nNo changes detected.")
            return False
        
        print(f"\n{'='*80}")
        
        # Ask for confirmation
        while True:
            response = input("Do you want to apply these changes? [y/ns/nd/d] (y=yes, ns=no+skip, nd=no+done, d=show diff again): ").lower().strip()
            
            if response in ['y', 'yes']:
                return "accept"
            elif response in ['ns', 'no+skip', 'skip']:
                return "reject_skip"  # Skip and leave undone for future attempts
            elif response in ['nd', 'no+done', 'done']:
                return "reject_done"  # Skip and mark as done
            elif response in ['n', 'no']:
                # For backward compatibility, ask for clarification
                print("Please clarify:")
                print("  ns = Skip and leave undone (can retry later)")
                print("  nd = Skip and mark done (won't retry)")
                continue
            elif response in ['d', 'diff']:
                # Show diff again
                for line in diff_output:
                    line = line.rstrip()
                    if line.startswith('---') or line.startswith('+++'):
                        print(f"\033[1m{line}\033[0m")  # Bold
                    elif line.startswith('-'):
                        print(f"\033[31m{line}\033[0m")  # Red
                    elif line.startswith('+'):
                        print(f"\033[32m{line}\033[0m")  # Green
                    elif line.startswith('@@'):
                        print(f"\033[36m{line}\033[0m")  # Cyan
                    else:
                        print(line)
                continue
            else:
                print("Please enter:")
                print("  y  = Accept changes")
                print("  ns = Skip and leave undone (can retry later)")
                print("  nd = Skip and mark done (won't retry)")
                print("  d  = Show diff again")
    
    def is_available(self) -> bool:
        """Check if AI README migration is available"""
        return True