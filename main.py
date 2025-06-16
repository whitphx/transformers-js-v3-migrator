import click
from src.migrator import TransformersJSMigrator


@click.command()
@click.option('--dry-run', is_flag=True, help='Run without making actual changes')
@click.option('--limit', default=10, help='Limit number of repositories to process')
@click.option('--token', help='Hugging Face token for API access')
@click.option('--org', help='Filter repositories by organization name')
@click.option('--repo-name', help='Filter repositories by name pattern (supports wildcards)')
@click.option('--exclude-org', multiple=True, help='Exclude repositories from these organizations')
def main(dry_run: bool, limit: int, token: str, org: str, repo_name: str, exclude_org: tuple):
    """Migrate Transformers.js v2 model repositories to v3"""
    migrator = TransformersJSMigrator(token=token, dry_run=dry_run)
    migrator.run_migration(
        limit=limit,
        org_filter=org,
        repo_name_filter=repo_name,
        exclude_orgs=list(exclude_org)
    )


if __name__ == "__main__":
    main()
