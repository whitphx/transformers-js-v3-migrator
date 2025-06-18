#!/usr/bin/env node

import { pipeline } from '@huggingface/transformers';
import { existsSync } from 'fs';
import { join } from 'path';

/**
 * Validation script for Transformers.js quantized models
 * 
 * Environment Variables:
 * - TASK_TYPE: The pipeline task type (e.g., 'automatic-speech-recognition')
 * - REPO_ID: The repository ID (e.g., 'Xenova/whisper-tiny.en')
 * - MODEL_FILENAME: The specific model file being validated
 * - CACHE_DIR: The cache directory containing the model files
 * - DTYPE: The quantization type (e.g., 'fp32', 'fp16', 'q8', 'uint8')
 */

// Configuration from environment variables
const config = {
    taskType: process.env.TASK_TYPE || 'feature-extraction',
    repoId: process.env.REPO_ID || '',
    modelFilename: process.env.MODEL_FILENAME || '',
    cacheDir: process.env.CACHE_DIR || '',
    dtype: process.env.DTYPE || 'fp32'
};

// Validate required configuration
function validateConfig() {
    const required = ['repoId', 'modelFilename', 'cacheDir'];
    const missing = required.filter(key => !config[key]);
    
    if (missing.length > 0) {
        console.error(`❌ Missing required environment variables: ${missing.map(k => k.toUpperCase()).join(', ')}`);
        process.exit(1);
    }
    
    // Verify cache directory exists
    const modelCacheDir = join(config.cacheDir, config.repoId);
    if (!existsSync(modelCacheDir)) {
        console.error(`❌ Model cache directory not found: ${modelCacheDir}`);
        process.exit(1);
    }
    
    console.log(`🔍 Validating model: ${config.modelFilename}`);
    console.log(`📁 Task type: ${config.taskType}`);
    console.log(`🏷️  Dtype: ${config.dtype}`);
    console.log(`📂 Cache directory: ${config.cacheDir}`);
}

async function validateModel() {
    try {
        validateConfig();
        
        console.log(`⏳ Loading pipeline for ${config.repoId}...`);
        
        // Create pipeline - this will test if the specific quantized model loads correctly
        const pipe = await pipeline(config.taskType, config.repoId, {
            dtype: config.dtype,
            device: 'cpu',  // Force CPU to avoid GPU issues
            local_files_only: true,  // Use local files only
            cache_dir: config.cacheDir
        });
        
        console.log('✅ Pipeline created successfully');
        console.log(`✅ Model ${config.modelFilename} validation passed`);
        
        // Cleanup
        if (pipe && typeof pipe.dispose === 'function') {
            await pipe.dispose();
            console.log('🧹 Pipeline disposed');
        }
        
        console.log(`🎉 Validation completed successfully for ${config.modelFilename}`);
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