from .readme_migration import ReadmeSamplesMigration
from ..migration_types import migration_registry

# Register all migrations
migration_registry.register(ReadmeSamplesMigration())

# TODO: Add other migration types as they are implemented
# from .model_binary_migration import ModelBinaryMigration
# from .config_migration import ConfigFilesMigration
# from .example_scripts_migration import ExampleScriptsMigration
# migration_registry.register(ModelBinaryMigration())
# migration_registry.register(ConfigFilesMigration())
# migration_registry.register(ExampleScriptsMigration())

__all__ = ['ReadmeSamplesMigration']