#!/usr/bin/env node

/**
 * CLI for running model validation with easier parameter passing
 */

import { spawn } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configure yargs
const argv = yargs(hideBin(process.argv))
    .usage('🔍 Transformers.js Model Validator CLI\n\nUsage: $0 [options]')
    .help('help')
    .alias('help', 'h')
    .option('model-path', {
        type: 'string',
        describe: 'Full path to the model directory',
        demandOption: true
    })
    .option('dtype', {
        type: 'string',
        describe: 'Quantization type to validate',
        demandOption: true
    })
    .option('task-type', {
        type: 'string',
        describe: 'Pipeline task type',
        demandOption: true
    })
    .option('debug', {
        type: 'boolean',
        describe: 'Enable debug logging',
        default: false
    })
    .strict()
    .parseSync();

function runValidation(config) {
    // Set up environment variables
    const env = {
        ...process.env,
        MODEL_PATH: config.modelPath,
        DTYPE: config.dtype,
        TASK_TYPE: config.taskType
    };
    
    if (config.debug) {
        env.DEBUG = 'true';
    }
    
    console.log(`🚀 Starting validation...`);
    console.log(`📁 Model path: ${config.modelPath}`);
    console.log(`🏷️  Dtype: ${config.dtype}`);
    console.log(`📁 Task type: ${config.taskType}`);
    console.log('');
    
    // Run the validation script
    const validationScript = join(__dirname, 'validate-model.js');
    const child = spawn('node', [validationScript], {
        env,
        stdio: 'inherit'
    });
    
    child.on('exit', (code) => {
        process.exit(code);
    });
    
    child.on('error', (error) => {
        console.error(`❌ Failed to start validation: ${error.message}`);
        process.exit(1);
    });
}

// Main execution - use yargs-parsed arguments
runValidation(argv);