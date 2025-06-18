/**
 * Utility functions for model validation
 */

/**
 * Extract dtype from model filename
 * @param {string} modelFilename - The model filename
 * @returns {string} The detected dtype
 */
export function extractDtypeFromFilename(modelFilename) {
    const filename = modelFilename.toLowerCase();
    
    if (filename.includes('_fp16')) return 'fp16';
    if (filename.includes('_q8')) return 'q8';
    if (filename.includes('_q4f16')) return 'q4f16';
    if (filename.includes('_q4')) return 'q4';
    if (filename.includes('_int8')) return 'int8';
    if (filename.includes('_uint8')) return 'uint8';
    if (filename.includes('_bnb4')) return 'bnb4';
    
    return 'fp32';  // default
}

/**
 * Infer task type from repository ID using common patterns
 * @param {string} repoId - The repository ID
 * @returns {string} The inferred task type
 */
export function inferTaskType(repoId) {
    const repoLower = repoId.toLowerCase();
    
    // Common task type patterns
    if (/whisper|speech|asr/.test(repoLower)) {
        return 'automatic-speech-recognition';
    }
    if (/bert|roberta|distilbert|classification/.test(repoLower)) {
        return 'text-classification';
    }
    if (/gpt|generation|text-generation/.test(repoLower)) {
        return 'text-generation';
    }
    if (/embedding|sentence|feature-extraction/.test(repoLower)) {
        return 'feature-extraction';
    }
    if (/translation|translate/.test(repoLower)) {
        return 'translation';
    }
    if (/summarization|summary/.test(repoLower)) {
        return 'summarization';
    }
    if (/question|qa/.test(repoLower)) {
        return 'question-answering';
    }
    if (/object-detection|detection|yolo/.test(repoLower)) {
        return 'object-detection';
    }
    if (/image-classification|vision|vit/.test(repoLower)) {
        return 'image-classification';
    }
    if (/segmentation/.test(repoLower)) {
        return 'image-segmentation';
    }
    
    // Default to feature-extraction as it's most generic
    return 'feature-extraction';
}

/**
 * Available quantization modes
 */
export const QUANTIZATION_MODES = [
    'fp16', 'q8', 'int8', 'uint8', 'q4', 'q4f16', 'bnb4'
];

/**
 * Validate if a dtype is supported
 * @param {string} dtype - The dtype to validate
 * @returns {boolean} Whether the dtype is supported
 */
export function isValidDtype(dtype) {
    return ['fp32', ...QUANTIZATION_MODES].includes(dtype);
}