import os
import logging
from typing import Optional
from anthropic import Anthropic


class AIReadmeMigrator:
    """AI-powered README migrator using external LLM API for Transformers.js v2 to v3 migration"""

    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for AI README migration")

        self.client = Anthropic(api_key=self.api_key)
        self.verbose = verbose
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
            validation_result = self._validate_migration(content, migrated_content)
            if validation_result == "trivial":
                self.logger.info(f"AI README migration skipped for {repo_id} - only trivial changes")
                return None  # Return None to indicate no changes needed (will be marked as SKIPPED)
            elif not validation_result:
                self.logger.error(f"AI README migration validation failed for {repo_id}")
                return None

            # Interactive mode: show changes and ask for confirmation
            if interactive:
                while True:
                    user_choice = self._show_changes_and_confirm(content, migrated_content, repo_id)
                    if user_choice == "accept":
                        break  # Continue with migration
                    elif user_choice == "reject_skip":
                        self.logger.info(f"User declined changes for {repo_id} - leaving undone for future attempts")
                        return "REJECT_SKIP"  # Special return value to indicate skip without marking done
                    elif user_choice == "reject_done":
                        self.logger.info(f"User declined changes for {repo_id} - marking as done")
                        return None  # Standard rejection, will be marked as completed/skipped
                    elif user_choice == "retry_llm":
                        self.logger.info(f"User requested LLM retry for {repo_id}")
                        # Ask the LLM again with a retry prompt
                        migrated_content = self._retry_with_llm(content, migrated_content, repo_id)
                        if not migrated_content:
                            self.logger.error(f"LLM retry failed for {repo_id}")
                            return None
                        # Continue the loop to show new changes
                        continue
                    elif user_choice == "edit_direct":
                        self.logger.info(f"User requested direct editing for {repo_id}")
                        # Allow user to edit the content directly
                        migrated_content = self._edit_content_directly(migrated_content, repo_id)
                        if not migrated_content:
                            self.logger.info(f"User cancelled direct editing for {repo_id}")
                            return None
                        # Continue the loop to show the edited changes
                        continue
                    else:
                        # This shouldn't happen, but handle gracefully
                        self.logger.info(f"User declined changes for {repo_id}")
                        return None

            self.logger.info(f"AI README migration completed successfully for {repo_id}")
            return migrated_content

        except Exception as e:
            if self.verbose:
                import traceback
                self.logger.error(f"Error during AI README migration for {repo_id}: {e}")
                self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            else:
                self.logger.error(f"Error during AI README migration for {repo_id}: {e}")
            return None

    def _create_migration_prompt(self, content: str, repo_id: str) -> str:
        """Create a detailed prompt for the AI README migration"""

        example_diff1 = """
## Example migration (v2 -> v3):

BEFORE:
'''''
```js
// npm i @xenova/transformers
import { pipeline } from '@xenova/transformers';

let url = 'https://example.com/audio.wav';
let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
let output = await transcriber(url);
```
'''''

AFTER:
'''''
```js
import { pipeline } from '@huggingface/transformers';

// Create speech recognition pipeline
const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');

// Transcribe audio from URL
const url = 'https://example.com/audio.wav';
const output = await transcriber(url);
```
'''''
"""

        example_diff2 = """
## Example migration (v2 -> v3):

BEFORE:
'''''

```js
// npm i @xenova/transformers
import { pipeline, cos_sim } from '@xenova/transformers';

// Create feature extraction pipeline
const extractor = await pipeline('feature-extraction', 'Xenova/jina-embeddings-v2-small-en',
    { quantized: false } // Comment out this line to use the quantized version
);

// Generate embeddings
const output = await extractor(
    ['How is the weather today?', 'What is the current weather like today?'],
    { pooling: 'mean' }
);

// Compute cosine similarity
console.log(cos_sim(output[0].data, output[1].data));  // 0.9399812684139274 (unquantized) vs. 0.9341121503699659 (quantized)
```
'''''

AFTER:
'''''

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

You can then use the model as follows:
```js
import { pipeline, cos_sim } from '@huggingface/transformers';

// Create feature extraction pipeline
const extractor = await pipeline('feature-extraction', 'Xenova/jina-embeddings-v2-small-en',
    { dtype: "fp32" } // Options: "fp32", "fp16", "q8", "q4"
);

// Generate embeddings
const output = await extractor(
    ['How is the weather today?', 'What is the current weather like today?'],
    { pooling: 'mean' }
);

// Compute cosine similarity
console.log(cos_sim(output[0].data, output[1].data));  // 0.9399812684139274 (unquantized) vs. 0.9341121503699659 (quantized)
```
'''''
"""

        example_diff3 = """
## Example migration (v2 -> v3):

BEFORE:
'''''
const classifier = await pipeline('image-classification', 'Xenova/convnext-tiny-224');
'''''

AFTER:
'''''
const classifier = await pipeline('image-classification', 'Xenova/convnext-tiny-224', {
    dtype: "fp32",  // Options: "fp32", "fp16", "q8", "q4"
});
'''''
"""

        prompt = f"""You are migrating a Transformers.js model repository README from v2 to v3. Your task is to update the README content while preserving its original structure and purpose.

## CRITICAL REQUIREMENTS:
1. **Output only the migrated README content** - NO wrapper text, explanations, or meta-commentary
2. **Preserve original structure** - Keep the same sections, formatting, and overall organization
3. **Minimal changes only** - Only update what's necessary for v3 compatibility
4. **PRESERVE FRONTMATTER** - Keep all YAML frontmatter (content between --- lines) exactly as-is

## Required Changes:
1. **Package name**: Change `@xenova/transformers` to `@huggingface/transformers`
2. **Installation instructions**: ALWAYS add installation instructions when there are code examples, even if they were missing before
3. **Add basic usage example**: If no code examples exist but the model can be used with Transformers.js, add a basic usage example
4. **Remove inline install comments**: Remove `// npm i @xenova/transformers` comments from code blocks because the installation instructions are already added as above
5. **Modern JavaScript**: Use `const` instead of `let` for variables that aren't reassigned
6. **Add semicolons**: Ensure statements end with semicolons where appropriate
7. **Keep code formats**: Keep the code formats such as white spaces, line breaks, etc. as is
8. **Update third argument of pipeline, the pipeline configuration**: Delete `{{ quantized: false }}` and add `{{ dtype: "fp32" }}` with a comment saying '// Options: "fp32", "fp16", "q8", "q4"'. Place the option in the next line after the pipeline creation line.

## Installation Section Template:
When adding installation instructions, use this format before the first code example:

```markdown
If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

You can then use the model as follows:
```

## Basic Usage Example Template:
If no code examples exist, add a basic usage example based on the model type. Use the repository ID from the prompt to create an appropriate example:

```js
import {{ pipeline }} from '@huggingface/transformers';

// Create the pipeline
const pipe = await pipeline('task-type', '{repo_id}');

// Use the model
const result = await pipe('input text or data');
console.log(result);
```

{example_diff1}

{example_diff2}

{example_diff3}

## STRICT GUIDELINES:
- **NEVER remove frontmatter** - Keep all YAML metadata between --- lines exactly as-is
- **ADD installation instructions** - Always add them before code examples if missing
- **ADD basic usage example** - If no code examples exist, add a simple usage example based on the model type
- DO NOT add explanatory text about what the code does beyond basic usage
- DO NOT move example outputs or change code structure
- DO NOT add sections that weren't in the original (except installation and basic usage)
- DO NOT add wrapper text like "Here is the migrated content"
- PRESERVE comments that are example outputs (like "// Found car at...")
- Keep the exact same markdown structure and sections
- Return ONLY the migrated README content, nothing else

## Repository: {repo_id}

## Original README Content:
{content}

## MIGRATED README (output only this):"""

        return prompt

    def _validate_migration(self, original: str, migrated: str):
        """Validate the AI README migration result

        Returns:
            True: Migration is valid and has meaningful changes
            False: Migration failed validation
            "trivial": Migration is valid but only has trivial changes
        """

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
        # Allow more flexibility for short READMEs that get expanded with installation instructions
        length_ratio = len(migrated) / len(original)
        original_length = len(original)

        # For very short READMEs (< 500 chars), allow up to 5x expansion (adding install instructions)
        # For medium READMEs (500-2000 chars), allow up to 3x expansion
        # For longer READMEs (> 2000 chars), allow up to 2x expansion
        if original_length < 500:
            max_ratio = 5.0
        elif original_length < 2000:
            max_ratio = 3.0
        else:
            max_ratio = 2.0

        if length_ratio < 0.5 or length_ratio > max_ratio:
            self.logger.error(f"README migration result length seems wrong: {length_ratio:.2f}x original (original: {original_length} chars, max allowed: {max_ratio}x)")
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

        # Check if changes are trivial (only whitespace/empty line changes)
        if self._are_changes_trivial(original, migrated):
            self.logger.info("README migration contains only trivial changes (whitespace/empty lines) - treating as completed")
            # Show the diff for preview when trivial changes are detected
            self._show_trivial_changes_diff(original, migrated)
            return "trivial"

        return True

    def _are_changes_trivial(self, original: str, migrated: str) -> bool:
        """Use LLM to determine if changes are trivial or meaningful"""

        # Quick check: if content is identical, it's trivial
        if original.strip() == migrated.strip():
            self.logger.debug("Changes are trivial - content is identical")
            return True

        # Quick check: if only whitespace differences, it's trivial
        original_no_whitespace = ''.join(original.split())
        migrated_no_whitespace = ''.join(migrated.split())
        if original_no_whitespace == migrated_no_whitespace:
            self.logger.debug("Changes are trivial - only whitespace differences")
            return True

        try:
            # Use LLM to determine if changes are trivial
            assessment_prompt = self._create_trivial_assessment_prompt(original, migrated)

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.0,  # Use deterministic output
                messages=[
                    {
                        "role": "user",
                        "content": assessment_prompt
                    }
                ]
            )

            assessment = response.content[0].text.strip().lower()

            # Parse the LLM response
            if assessment.startswith("trivial"):
                self.logger.debug("LLM determined changes are trivial")
                return True
            elif assessment.startswith("meaningful"):
                self.logger.debug("LLM determined changes are meaningful")
                return False
            else:
                # If unclear response, err on the side of meaningful
                self.logger.warning(f"Unclear LLM assessment: {assessment}, treating as meaningful")
                return False

        except Exception as e:
            # If LLM call fails, err on the side of meaningful changes
            self.logger.warning(f"Failed to assess changes with LLM: {e}, treating as meaningful")
            return False

    def _create_trivial_assessment_prompt(self, original: str, migrated: str) -> str:
        """Create a prompt for the LLM to assess if changes are trivial or meaningful"""

        import difflib

        # Generate a unified diff
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile="original",
            tofile="migrated",
            lineterm=""
        ))

        diff_text = ''.join(diff_lines) if diff_lines else "No differences detected"

        prompt = f"""You are evaluating whether changes to a README file are trivial or meaningful for a Transformers.js v2 to v3 migration.

TRIVIAL changes include:
- Only whitespace, formatting, or empty line changes
- Minor punctuation or capitalization changes
- Reordering of identical content
- Changes that don't affect functionality or user understanding

MEANINGFUL changes include:
- Adding or updating package names (@xenova/transformers → @huggingface/transformers)
- Adding installation instructions or usage examples
- Adding or updating code configuration (like {{ dtype: "fp32" }})
- Adding comments that explain options or usage
- Any changes that improve user understanding or functionality
- Any changes related to v3 migration requirements

Here are the changes:

```diff
{diff_text}
```

Respond with exactly one word: either "TRIVIAL" or "MEANINGFUL" based on whether these changes provide value to users migrating from Transformers.js v2 to v3."""

        return prompt

    def _show_trivial_changes_diff(self, original: str, migrated: str):
        """Show diff for trivial changes that are being filtered out"""
        import difflib

        print(f"\n{'='*80}")
        print(f"TRIVIAL CHANGES DETECTED - SKIPPING MIGRATION")
        print(f"{'='*80}")
        print(f"The AI suggested changes, but they are only trivial (whitespace/empty lines).")
        print(f"No pull request will be created for these changes.")
        print(f"{'='*80}")

        # Generate and display diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile="README.md (original)",
            tofile="README.md (ai-suggested)",
            lineterm=""
        )

        diff_output = list(diff)
        if diff_output:
            print("\nTrivial changes that were filtered out:")
            line_count = 0
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

                line_count += 1
                # Limit output to prevent overwhelming the console
                if line_count > 50:
                    print(f"\033[33m... (diff truncated after 50 lines)\033[0m")
                    break
        else:
            print("\nNo visible differences in the diff output.")

        print(f"\n{'='*80}")
        print(f"Migration marked as SKIPPED - repository will be marked as completed.")
        print(f"{'='*80}\n")

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
            response = input("Do you want to apply these changes? [y/ns/nd/d/r/e] (y=yes, ns=no+skip, nd=no+done, d=diff, r=retry LLM, e=edit): ").lower().strip()

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
            elif response in ['r', 'retry']:
                return "retry_llm"  # Ask the LLM again
            elif response in ['e', 'edit']:
                return "edit_direct"  # Allow direct editing
            else:
                print("Please enter:")
                print("  y  = Accept changes")
                print("  ns = Skip and leave undone (can retry later)")
                print("  nd = Skip and mark done (won't retry)")
                print("  d  = Show diff again")
                print("  r  = Ask LLM to try again")
                print("  e  = Edit the content directly")

    def _retry_with_llm(self, original: str, previous_attempt: str, repo_id: str) -> Optional[str]:
        """Ask the LLM to try again with feedback about the previous attempt"""
        try:
            self.logger.info(f"Retrying LLM migration for {repo_id} with feedback")

            # Create a retry prompt that includes the previous attempt
            retry_prompt = self._create_retry_prompt(original, previous_attempt, repo_id)

            # Call Claude API with retry prompt
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.2,  # Slightly higher temperature for variety
                messages=[
                    {
                        "role": "user",
                        "content": retry_prompt
                    }
                ]
            )

            new_migrated_content = response.content[0].text.strip()

            # Validate the retry response
            validation_result = self._validate_migration(original, new_migrated_content)
            if validation_result == "trivial":
                self.logger.info(f"LLM retry for {repo_id} resulted in trivial changes")
                return None
            elif not validation_result:
                self.logger.error(f"LLM retry validation failed for {repo_id}")
                return None

            self.logger.info(f"LLM retry completed successfully for {repo_id}")
            return new_migrated_content

        except Exception as e:
            if self.verbose:
                import traceback
                self.logger.error(f"Error during LLM retry for {repo_id}: {e}")
                self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            else:
                self.logger.error(f"Error during LLM retry for {repo_id}: {e}")
            return None

    def _create_retry_prompt(self, original: str, previous_attempt: str, repo_id: str) -> str:
        """Create a retry prompt that asks the LLM to improve on the previous attempt"""
        prompt = f"""You are retrying a README migration for a Transformers.js repository. The user was not satisfied with your previous attempt and wants you to try a different approach.

## CRITICAL REQUIREMENTS (same as before):
1. **Output only the migrated README content** - NO wrapper text, explanations, or meta-commentary
2. **Preserve original structure** - Keep the same sections, formatting, and overall organization
3. **Minimal changes only** - Only update what's necessary for v3 compatibility
4. **PRESERVE FRONTMATTER** - Keep all YAML frontmatter (content between --- lines) exactly as-is

## Required Changes:
1. **Package name**: Change `@xenova/transformers` to `@huggingface/transformers`
2. **Installation instructions**: Add installation instructions when there are code examples
3. **Modern JavaScript**: Use `const` instead of `let`, add semicolons
4. **Update pipeline configuration**: Replace `{{ quantized: false }}` with `{{ dtype: "fp32" }}` and add comment about options

## Repository: {repo_id}

## Original README Content:
{original}

## Previous Attempt (user was not satisfied):
{previous_attempt}

## Instructions for Retry:
- Try a different approach than the previous attempt
- Consider adding more/less content as needed
- Focus on making the changes more substantial and helpful
- Ensure installation instructions are clear and prominent
- Make sure code examples are complete and functional

## NEW MIGRATED README (output only this):"""

        return prompt

    def _edit_content_directly(self, content: str, repo_id: str) -> Optional[str]:
        """Allow user to edit the content directly using their preferred editor"""
        import tempfile
        import subprocess
        import os

        try:
            print(f"\n{'='*80}")
            print(f"DIRECT EDITING MODE for {repo_id}")
            print(f"{'='*80}")
            print("Opening the content in your default editor...")
            print("Make your changes and save the file, then close the editor to continue.")
            print(f"{'='*80}")

            # Create a temporary file with the content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Try to get the user's preferred editor
                editor = os.environ.get('EDITOR') or os.environ.get('VISUAL')

                if not editor:
                    # Try common editors in order of preference
                    for potential_editor in ['nano', 'vim', 'vi', 'code', 'notepad']:
                        if subprocess.run(['which', potential_editor], capture_output=True).returncode == 0:
                            editor = potential_editor
                            break

                if not editor:
                    print("No suitable editor found. Please set the EDITOR environment variable.")
                    return None

                # Open the editor
                result = subprocess.run([editor, tmp_file_path])

                if result.returncode != 0:
                    print(f"Editor exited with error code {result.returncode}")
                    return None

                # Read the edited content
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    edited_content = f.read()

                # Check if content was actually changed
                if edited_content.strip() == content.strip():
                    print("No changes detected in the edited content.")
                    return None

                print("Content has been edited successfully!")
                return edited_content

            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass

        except Exception as e:
            self.logger.error(f"Error during direct editing for {repo_id}: {e}")
            return None

    def is_available(self) -> bool:
        """Check if AI README migration is available"""
        return True
