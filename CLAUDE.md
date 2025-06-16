# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered migration agent that automates the migration of Transformers.js v2 model repositories to v3. The tool searches Hugging Face Hub for repositories using Transformers.js library, analyzes their code, and creates pull requests with migration changes.

## Key Commands

### Running the Migration Tool
```bash
# Preview repositories that would be processed (safe mode)
uv run python main.py migrate --org whitphx --preview

# Dry run with progress tracking but no actual changes
uv run python main.py migrate --org whitphx --limit 5 --dry-run

# List all migration sessions
uv run python main.py sessions

# Check detailed status of a specific session
uv run python main.py status <session_id>
```

### Development Setup
```bash
# Install dependencies
uv sync

# Run with specific filters
uv run python main.py migrate --org whitphx --repo-name "dummy-*" --dry-run
```

### Testing
Always use the test repository `whitphx/dummy-transformerjs-model-000` for testing and development. This repository contains v2 Transformers.js code and is safe to use for testing the migration logic.

## Architecture Overview

### Core Components

- **TransformersJSMigrator** (`src/migrator.py`): Main orchestrator that coordinates the entire migration process
- **Migration Types Framework** (`src/migration_types.py`): Pluggable architecture for different migration types using abstract base classes and a registry pattern
- **Session Management** (`src/session_manager.py`): Persistent progress tracking with resume capability, stores session data in `.sessions/` directory
- **HF Hub Integration** (`src/git_operations.py`): Uses HuggingFace Hub API methods instead of Git operations for downloading and uploading files

### Migration Types System

The tool uses a pluggable migration system where each migration type inherits from `BaseMigration`:

- **ReadmeSamplesMigration**: AI-powered migration of README.md files from Transformers.js v2 to v3
  - Uses Anthropic Claude API for intelligent code transformation
  - Falls back to rule-based migration if AI is unavailable
  - Handles installation sections, modern JavaScript patterns, and improved comments
- **Future migrations**: Model binaries, config files, example scripts (framework ready)

Each migration type creates separate pull requests and tracks progress independently.

### Session Management

Sessions are identified by a hash of search parameters (org, repo-name, exclude-orgs). Key features:
- **Global skip logic**: Repositories processed in ANY session are permanently skipped
- **Per-migration tracking**: Each repo tracks status of individual migration types
- **Resume capability**: Can resume failed migrations or specific migration types

### Operating Modes

- **Preview**: Lists repositories without processing or tracking (`--preview`)
- **Dry Run**: Downloads repos, checks migrations, tracks progress but makes no changes (`--dry-run`)
- **Normal**: Full migration with file uploads and PR creation

## Environment Variables

- `HF_TOKEN`: Hugging Face token for API access (can also be passed via `--token`)
- `ANTHROPIC_API_KEY`: Anthropic API key for AI-powered README migrations (optional, falls back to rule-based migration)

## File Operations

The tool uses HuggingFace Hub's native API methods:
- `snapshot_download()` for downloading repositories
- `CommitOperationAdd` + `create_commit()` for uploading files
- `create_discussion()` for creating pull requests

No Git dependencies required - all operations go through HF Hub API.

## Session Data Structure

Sessions are stored as JSON files in `.sessions/` with the structure:
```json
{
  "repos": {
    "org/repo": {
      "status": "completed",
      "migrations": {
        "readme_samples": {
          "status": "completed",
          "pr_url": "...",
          "files_modified": ["README.md"]
        }
      }
    }
  }
}
```

## Adding New Migration Types

1. Create a new class inheriting from `BaseMigration` in `src/migrations/`
2. Implement required methods: `migration_type`, `description`, `is_applicable`, `apply_migration`
3. Register the migration in `src/migrations/__init__.py`
4. Add the migration type to the `MigrationType` enum

The framework will automatically handle session tracking, PR creation, and progress management.