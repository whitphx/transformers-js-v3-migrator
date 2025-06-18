#!/usr/bin/env node

import { pipeline, env } from '@huggingface/transformers';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import winston from 'winston';

// Allow local models to be loaded and disable remote access
env.allowLocalModels = true;
env.allowRemoteModels = false;

/**
 * Validation script for Transformers.js quantized models
 *
 * Environment Variables:
 * - MODEL_PATH: Full path to the model directory
 * - DTYPE: The quantization type (e.g., 'fp32', 'fp16', 'q8', 'uint8')
 * - TASK_TYPE: The pipeline task type
 * - DEBUG: Enable debug mode for development
 */

// Configuration from environment variables
const config = {
    modelPath: process.env.MODEL_PATH,
    dtype: process.env.DTYPE,
    taskType: process.env.TASK_TYPE,
    debug: process.env.DEBUG === 'true' || process.env.DEBUG === '1'
};

// Configure logger
const logLevel = config.debug ? 'debug' : 'info';
const logger = winston.createLogger({
    level: logLevel,
    format: winston.format.combine(
        winston.format.simple()  // Remove colorize for better subprocess capture
    ),
    transports: [
        new winston.transports.Console({
            stderrLevels: ['error']  // Send errors to stderr, info/debug to stdout
        })
    ]
});

// Write validation result to stdout as the last line
function writeResult(result) {
    console.log(`VALIDATION_RESULT:${JSON.stringify(result)}`);
}

// Validate required configuration
function validateConfig() {
    const required = ['modelPath', 'dtype', 'taskType'];
    const missing = required.filter(key => !config[key]);

    if (missing.length > 0) {
        logger.error(`❌ Missing: ${missing.map(k => k.toUpperCase()).join(', ')}`);
        writeResult({ success: false, error: `Missing required parameters: ${missing.join(', ')}` });
        process.exit(1);
    }

    if (!existsSync(config.modelPath)) {
        logger.error(`❌ Model directory not found: ${config.modelPath}`);
        writeResult({ success: false, error: `Model directory not found: ${config.modelPath}` });
        process.exit(1);
    }

    logger.info(`🔍 Validating ${config.modelPath} (${config.dtype})`);
    logger.debug(`📁 Model path: ${config.modelPath}`);
    logger.debug(`📋 Task type: ${config.taskType}`);
}

async function validateModel() {
    try {
        validateConfig();

        // Create pipeline with full model path and specified dtype
        logger.debug(`⏳ Loading pipeline with dtype: ${config.dtype}`);
        const pipe = await pipeline(config.taskType, config.modelPath, {
            dtype: config.dtype,
            device: 'cpu',
            local_files_only: true
        });

        logger.info(`✅ Validation passed for ${config.modelPath} (${config.dtype})`);
        
        // Write success result to stdout
        writeResult({ 
            success: true, 
            modelPath: config.modelPath, 
            dtype: config.dtype,
            taskType: config.taskType
        });
        
        // Skip cleanup to avoid ONNX runtime mutex issues
        // The validation succeeded, so we can exit immediately
        process.exit(0);

    } catch (error) {
        logger.error(`❌ Model validation failed: ${error.message}`);
        logger.debug('📋 Error details:', error);
        
        // Write failure result to stdout
        writeResult({ 
            success: false, 
            error: error.message,
            modelPath: config.modelPath, 
            dtype: config.dtype 
        });
        
        process.exit(1);
    }
}

// Handle unhandled errors
process.on('unhandledRejection', (error) => {
    logger.error('💥 Unhandled error in validation:', error);
    writeResult({ 
        success: false, 
        error: `Unhandled error: ${error.message}`,
        modelPath: config.modelPath, 
        dtype: config.dtype 
    });
    process.exit(1);
});

// Run validation
validateModel();
