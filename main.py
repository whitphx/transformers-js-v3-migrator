import click
import os
from src.migrator import TransformersJSMigrator


@click.group()
def cli():
    """Transformers.js v2 to v3 Migration Tool"""
    pass


@cli.command()
@click.option('--dry-run', is_flag=True, help='Run without making actual changes but track progress')
@click.option('--preview', is_flag=True, help='Preview mode - no changes, no tracking')
@click.option('--local', is_flag=True, help='Local mode - apply changes locally but skip commits/pushes')
@click.option('--limit', default=10, help='Limit number of repositories to process')
@click.option('--token', help='Hugging Face token for API access (or set HF_TOKEN env var)')
@click.option('--repo', help='Exact repository name (format: org/repo-name)')
@click.option('--repo-search', help='Search repositories by name pattern')
@click.option('--author', help='Filter repositories by author/organization name')
@click.option('--exclude-org', multiple=True, help='Exclude repositories from these organizations')
@click.option('--resume', is_flag=True, help='Resume from existing session')
@click.option('--non-interactive', is_flag=True, help='Run without user prompts (auto-approve AI changes)')
@click.option('--verbose', is_flag=True, help='Enable verbose mode with detailed error tracebacks')
@click.option('--save-debug-models', is_flag=True, help='Save quantized models to debug directory for development')
@click.option('--debug-models-dir', help='Directory to save debug models (default: ./debug_models)')
@click.option('--migration-types', multiple=True, help='Run only specific migration types (e.g., readme_samples, model_binaries)')
@click.option('--list-migrations', is_flag=True, help='List available migration types and exit')
def migrate(dry_run: bool, preview: bool, local: bool, limit: int, token: str, repo: str, repo_search: str, author: str, exclude_org: tuple, resume: bool, non_interactive: bool, verbose: bool, save_debug_models: bool, debug_models_dir: str, migration_types: tuple, list_migrations: bool):
    """Run the migration process"""
    
    # Handle --list-migrations option
    if list_migrations:
        from src.migration_types import migration_registry
        import src.migrations  # Import to register migrations
        
        click.echo("\n🔧 Available Migration Types:")
        click.echo("=" * 50)
        
        all_migrations = migration_registry.get_all_migrations()
        for migration_class in all_migrations:
            migration_type = migration_class.migration_type.value
            # Create instance to get description
            instance = migration_class.__class__(verbose=False)
            description = instance.description
            
            click.echo(f"\n📋 {migration_type}")
            click.echo(f"   {description}")
        
        click.echo(f"\n💡 Usage Examples:")
        click.echo(f"   # Run only README migrations")
        click.echo(f"   python main.py migrate --migration-types readme_samples --repo whitphx/dummy-transformerjs-model-000")
        click.echo(f"   ")
        click.echo(f"   # Run only model binary migrations")
        click.echo(f"   python main.py migrate --migration-types model_binaries --repo whitphx/dummy-transformerjs-model-000")
        click.echo(f"   ")
        click.echo(f"   # Run multiple specific migrations")
        click.echo(f"   python main.py migrate --migration-types readme_samples --migration-types model_binaries --repo whitphx/dummy-transformerjs-model-000")
        click.echo("")
        return
    
    # Check for mutually exclusive mode options
    mode_flags = [dry_run, preview, local]
    if sum(mode_flags) > 1:
        click.echo("Error: Cannot use multiple mode flags (--dry-run, --preview, --local) at the same time")
        return
    
    # Check for mutually exclusive search options
    search_flags = [repo, repo_search, author]
    search_flag_count = sum(1 for flag in search_flags if flag)
    if search_flag_count > 1:
        click.echo("Error: Cannot use multiple search options (--repo, --repo-search, --author) at the same time")
        return
    
    # Validate migration types if specified
    if migration_types:
        from src.migration_types import migration_registry
        import src.migrations  # Import to register migrations
        
        # Get available migration types
        all_migrations = migration_registry.get_all_migrations()
        available_types = {migration.migration_type.value for migration in all_migrations}
        
        # Check if all specified types are valid
        invalid_types = set(migration_types) - available_types
        if invalid_types:
            click.echo(f"Error: Invalid migration types: {', '.join(invalid_types)}")
            click.echo(f"Available types: {', '.join(sorted(available_types))}")
            click.echo("Use --list-migrations to see all available migration types")
            return
        
        click.echo(f"🎯 Running only specified migrations: {', '.join(migration_types)}")
    
    # Get token from CLI argument or environment variable
    hf_token = token or os.getenv('HF_TOKEN')
    
    # Determine mode
    if preview:
        mode = "preview"
    elif dry_run:
        mode = "dry_run"
    elif local:
        mode = "local"
    else:
        mode = "normal"
    
    migrator = TransformersJSMigrator(
        token=hf_token, 
        mode=mode, 
        verbose=verbose,
        save_debug_models=save_debug_models,
        debug_models_dir=debug_models_dir,
        migration_types_filter=list(migration_types) if migration_types else None
    )
    migrator.run_migration(
        limit=limit,
        exact_repo=repo,
        repo_search=repo_search,
        author_filter=author,
        exclude_orgs=list(exclude_org),
        resume=resume,
        interactive=not non_interactive
    )


@cli.command()
def sessions():
    """List all migration sessions"""
    from src.session_manager import SessionManager
    
    session_manager = SessionManager()
    sessions_list = session_manager.list_sessions()
    
    if not sessions_list:
        click.echo("No sessions found.")
        return
    
    click.echo("\nMigration Sessions:")
    click.echo("=" * 80)
    
    for session in sessions_list:
        stats = session.get("stats", {})
        config = session.get("config") or {}
        
        click.echo(f"\nSession ID: {session['session_id']}")
        click.echo(f"Created: {session.get('created_at', 'Unknown')}")
        click.echo(f"Updated: {session.get('updated_at', 'Unknown')}")
        # Build config display string
        config_parts = []
        if config.get('exact_repo'):
            config_parts.append(f"exact_repo={config['exact_repo']}")
        if config.get('repo_search'):
            config_parts.append(f"repo_search={config['repo_search']}")
        if config.get('author_filter'):
            config_parts.append(f"author={config['author_filter']}")
        if not config_parts:
            config_parts.append("all_repos")
        
        click.echo(f"Config: {', '.join(config_parts)}")
        click.echo(f"Progress: {stats.get('completed', 0)}/{stats.get('total', 0)} completed, {stats.get('failed', 0)} failed")
        
        if session.get("recent_failures"):
            click.echo(f"Recent failures: {len(session['recent_failures'])}")


@cli.command()
@click.argument('session_id')
def status(session_id: str):
    """Show detailed status of a specific session"""
    from src.session_manager import SessionManager
    
    session_manager = SessionManager()
    summary = session_manager.get_session_summary(session_id)
    
    if not summary.get("stats"):
        click.echo(f"Session {session_id} not found.")
        return
    
    stats = summary["stats"]
    config = summary.get("config") or {}
    
    click.echo(f"\nSession: {session_id}")
    click.echo("=" * 50)
    click.echo(f"Created: {summary.get('created_at')}")
    click.echo(f"Updated: {summary.get('updated_at')}")
    click.echo(f"\nConfiguration:")
    click.echo(f"  Limit: {config.get('limit')}")
    click.echo(f"  Exact repo: {config.get('exact_repo', 'none')}")
    click.echo(f"  Repo search: {config.get('repo_search', 'none')}")
    click.echo(f"  Author filter: {config.get('author_filter', 'none')}")
    click.echo(f"  Excluded orgs: {', '.join(config.get('exclude_orgs', []))}")
    click.echo(f"  Dry run: {config.get('dry_run', False)}")
    
    click.echo(f"\nProgress:")
    click.echo(f"  Total repositories: {stats.get('total', 0)}")
    click.echo(f"  Completed: {stats.get('completed', 0)}")
    click.echo(f"  Failed: {stats.get('failed', 0)}")
    click.echo(f"  Pending: {stats.get('pending', 0)}")
    
    if summary.get("recent_failures"):
        click.echo(f"\nRecent Failures:")
        for failure in summary["recent_failures"]:
            click.echo(f"  - {failure['repo_id']}: {failure['error']}")


@cli.command()
@click.option('--repo', required=True, help='Repository to test (format: org/repo-name)')
@click.option('--migration-type', required=True, help='Migration type to test (use --list to see available)')
@click.option('--list', 'list_types', is_flag=True, help='List available migration types')
@click.option('--dry-run', is_flag=True, help='Run without making actual changes')
@click.option('--save-debug-models', is_flag=True, help='Save quantized models for debugging')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def debug(repo: str, migration_type: str, list_types: bool, dry_run: bool, save_debug_models: bool, verbose: bool):
    """Debug/test specific migration types on a single repository"""
    
    # Handle --list option
    if list_types:
        from src.migration_types import migration_registry
        import src.migrations  # Import to register migrations
        
        click.echo("\n🔧 Available Migration Types for Debug:")
        click.echo("=" * 50)
        
        all_migrations = migration_registry.get_all_migrations()
        for migration_class in all_migrations:
            migration_type_name = migration_class.migration_type.value
            # Create instance to get description
            instance = migration_class.__class__(verbose=False)
            description = instance.description
            
            click.echo(f"\n📋 {migration_type_name}")
            click.echo(f"   {description}")
        
        click.echo(f"\n💡 Debug Usage Example:")
        click.echo(f"   python main.py debug --repo whitphx/dummy-transformerjs-model-000 --migration-type model_binaries --dry-run --save-debug-models")
        click.echo("")
        return
    
    # Validate migration type
    from src.migration_types import migration_registry
    import src.migrations  # Import to register migrations
    
    all_migrations = migration_registry.get_all_migrations()
    available_types = {migration.migration_type.value for migration in all_migrations}
    
    if migration_type not in available_types:
        click.echo(f"❌ Invalid migration type: {migration_type}")
        click.echo(f"Available types: {', '.join(sorted(available_types))}")
        click.echo("Use --list to see all available migration types")
        return
    
    # Get token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        click.echo("❌ HF_TOKEN environment variable is required")
        return
    
    # Determine mode
    mode = "dry_run" if dry_run else "local"  # Use local mode for debug to avoid creating PRs
    
    click.echo(f"\n🐛 Debug Mode - Testing Migration")
    click.echo("=" * 40)
    click.echo(f"📂 Repository: {repo}")
    click.echo(f"🔧 Migration: {migration_type}")
    click.echo(f"⚙️  Mode: {mode}")
    click.echo(f"💾 Save debug models: {save_debug_models}")
    click.echo("=" * 40)
    click.echo("")
    
    # Create migrator with debug settings
    migrator = TransformersJSMigrator(
        token=hf_token,
        mode=mode,
        verbose=verbose,
        save_debug_models=save_debug_models,
        debug_models_dir="./debug_models",
        migration_types_filter=[migration_type]
    )
    
    # Run migration on single repository
    migrator.run_migration(
        limit=1,
        exact_repo=repo,
        interactive=True
    )


if __name__ == "__main__":
    cli()
