#!/usr/bin/env node

import { pipeline, env } from '@huggingface/transformers';
import { existsSync } from 'fs';
import { join, dirname } from 'path';

// Allow local models to be loaded and disable remote access
env.allowLocalModels = true;
env.allowRemoteModels = false;

/**
 * Validation script for Transformers.js quantized models
 *
 * Environment Variables:
 * - MODEL_BASE_DIR: Base directory containing model repositories
 * - MODEL_ID: The model ID (e.g., 'Xenova/whisper-tiny.en')
 * - DTYPE: The quantization type (e.g., 'fp32', 'fp16', 'q8', 'uint8')
 * - TASK_TYPE: The pipeline task type (auto-detected)
 * - DEBUG: Enable debug mode for development
 */

// Configuration from environment variables
const config = {
    modelBaseDir: process.env.MODEL_BASE_DIR,
    modelId: process.env.MODEL_ID,
    dtype: process.env.DTYPE,
    taskType: process.env.TASK_TYPE,
    debug: process.env.DEBUG === 'true' || process.env.DEBUG === '1'
};

// Validate required configuration
function validateConfig() {
    const required = ['modelBaseDir', 'modelId', 'dtype', 'taskType'];
    const missing = required.filter(key => !config[key]);

    if (missing.length > 0) {
        console.error(`❌ Missing: ${missing.map(k => k.toUpperCase()).join(', ')}`);
        process.exit(1);
    }

    if (!existsSync(config.modelBaseDir)) {
        console.error(`❌ Directory not found: ${config.modelBaseDir}`);
        process.exit(1);
    }

    const modelDir = join(config.modelBaseDir, config.modelId);
    if (!existsSync(modelDir)) {
        console.error(`❌ Model not found: ${modelDir}`);
        process.exit(1);
    }

    console.log(`🔍 Validating ${config.modelId} (${config.dtype})`);
}

async function validateModel() {
    try {
        validateConfig();

        // Set the model base directory as the local model path
        env.localModelPath = config.modelBaseDir;

        // Create pipeline with specified dtype
        const pipe = await pipeline(config.taskType, config.modelId, {
            dtype: config.dtype,
            device: 'cpu',
            local_files_only: true
        });

        console.log(`✅ Validation passed for ${config.modelId} (${config.dtype})`);
        
        // Skip cleanup to avoid ONNX runtime mutex issues
        // The validation succeeded, so we can exit immediately
        process.exit(0);

    } catch (error) {
        console.error(`❌ Model validation failed: ${error.message}`);
        console.error('📋 Error details:', error);
        process.exit(1);
    }
}

// Handle unhandled errors
process.on('unhandledRejection', (error) => {
    console.error('💥 Unhandled error in validation:', error);
    process.exit(1);
});

// Run validation
validateModel();
