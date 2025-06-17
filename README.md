# Transformers.js v2 to v3 Migrator

AI agent for migrating Transformers.js v2 model repositories to v3.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Modern Python package manager
- Python 3.12+
- Git

## Setup

1. **Install uv (if not already installed):**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or via package managers
   brew install uv          # macOS
   pip install uv           # Any platform
   ```

2. **Clone the repository with submodules:**
   ```bash
   git clone --recursive <repository-url>
   cd transformers-js-v3-migrator
   
   # Or if already cloned:
   git submodule update --init
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Set up environment variables:**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export ANTHROPIC_API_KEY="your_anthropic_api_key"  # For AI-powered README migration
   ```

## Usage

```bash
# Basic migration
uv run python main.py migrate --org whitphx

# With options
uv run python main.py migrate --org whitphx --limit 5 --dry-run

# Resume previous session
uv run python main.py migrate --resume

# Check session status
uv run python main.py sessions list
uv run python main.py status <session_id>
```

## Features

- **README Migration**: AI-powered migration of sample code from v2 to v3
- **Model Binary Migration**: Quantization of ONNX models (q4, fp16 variants)
- **Session Management**: Resume interrupted migrations
- **Multiple Migration Types**: Separate PRs for different migration types
- **Interactive Mode**: Review changes before applying

## Dependencies

The project uses the transformers.js repository as a submodule to access quantization scripts. Dependencies for model quantization are automatically managed using `uv run --with-requirements` for isolated execution, ensuring no conflicts with the main environment.

## Architecture

- `src/migrator.py`: Main orchestrator
- `src/migrations/`: Individual migration implementations
- `src/session_manager.py`: Progress tracking and resume functionality
- `transformers-js/`: Git submodule for quantization scripts