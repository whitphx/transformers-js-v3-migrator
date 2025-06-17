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
def migrate(dry_run: bool, preview: bool, local: bool, limit: int, token: str, repo: str, repo_search: str, author: str, exclude_org: tuple, resume: bool, non_interactive: bool, verbose: bool):
    """Run the migration process"""
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
    
    migrator = TransformersJSMigrator(token=hf_token, mode=mode, verbose=verbose)
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


if __name__ == "__main__":
    cli()
