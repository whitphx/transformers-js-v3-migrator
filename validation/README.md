# Transformers.js Model Validation

This Node.js package provides validation utilities for Transformers.js quantized models. It's used by the migration tool to verify that quantized model files can be successfully loaded by Transformers.js pipelines.

## Installation

```bash
npm install
```

## Usage

### Environment Variables

The validation script uses environment variables for configuration:

- `TASK_TYPE`: The pipeline task type (e.g., 'automatic-speech-recognition')
- `REPO_ID`: The repository ID (e.g., 'Xenova/whisper-tiny.en')  
- `MODEL_FILENAME`: The specific model file being validated
- `CACHE_DIR`: The cache directory containing the model files
- `DTYPE`: The quantization type (e.g., 'fp32', 'fp16', 'q8', 'uint8')

### Running Validation

```bash
# Set environment variables and run validation
export TASK_TYPE="automatic-speech-recognition"
export REPO_ID="Xenova/whisper-tiny.en"
export MODEL_FILENAME="decoder_model_uint8.onnx"
export CACHE_DIR="/path/to/cache"
export DTYPE="uint8"

npm run validate
```

### Example Usage from Python

```python
import subprocess
import os

env = os.environ.copy()
env.update({
    'TASK_TYPE': 'automatic-speech-recognition',
    'REPO_ID': 'Xenova/whisper-tiny.en',
    'MODEL_FILENAME': 'decoder_model_uint8.onnx',
    'CACHE_DIR': '/path/to/cache',
    'DTYPE': 'uint8'
})

result = subprocess.run(
    ["npm", "run", "validate"],
    cwd="validation",
    capture_output=True,
    text=True,
    timeout=60,
    env=env
)

if result.returncode == 0:
    print("✅ Model validation passed")
else:
    print("❌ Model validation failed")
    print(result.stderr)
```

## Supported Tasks

The validation automatically infers task types based on repository patterns:

- `automatic-speech-recognition`: whisper, speech, asr models
- `text-classification`: bert, roberta, distilbert models
- `text-generation`: gpt, generation models
- `feature-extraction`: embedding, sentence models
- `translation`: translation, translate models
- `image-classification`: vision, vit models
- `object-detection`: yolo, detection models
- And more...

## Supported Quantization Types

- `fp32`: 32-bit floating point (default)
- `fp16`: 16-bit floating point
- `q8`: 8-bit quantization
- `int8`: 8-bit integer
- `uint8`: 8-bit unsigned integer  
- `q4`: 4-bit quantization
- `q4f16`: 4-bit quantization with 16-bit fallback
- `bnb4`: BitsAndBytes 4-bit quantization

## Project Structure

```
validation/
├── package.json          # Node.js project configuration
├── README.md             # This file
└── src/
    ├── validate-model.js  # Main validation script
    └── utils.js          # Utility functions
```

## Error Handling

The validation script provides clear error messages and appropriate exit codes:

- Exit code 0: Validation successful
- Exit code 1: Validation failed (missing config, model load failure, etc.)

All errors are logged with descriptive messages and emojis for easy identification.