import os
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Optional
from ..migration_types import BaseMigration, MigrationType, MigrationResult, MigrationStatus


class ModelBinaryMigration(BaseMigration):
    """Migration for converting and quantizing ONNX model binaries"""
    
    # Define available quantization modes
    QUANTIZATION_MODES = ("fp16", "q8", "int8", "uint8", "q4", "q4f16", "bnb4")
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        # Get the path to the transformers.js submodule
        self.project_root = Path(__file__).parent.parent.parent
        self.transformers_js_path = self.project_root / "transformers-js"
        
        # Track quantization details for PR description
        self.quantization_details = {
            "added_for_missing": [],     # Modes added because files were missing
            "added_for_invalid": [],     # Modes added because files were invalid  
            "removed_invalid": [],       # Modes removed because invalid files couldn't be regenerated
            "models_processed": []       # List of base models that were processed
        }
    
    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.MODEL_BINARIES
    
    @property
    def description(self) -> str:
        modes_str = ", ".join(self.QUANTIZATION_MODES)
        return f"Add missing quantized ONNX model variants ({modes_str}) for Transformers.js v3"
    
    def is_applicable(self, repo_path: str, repo_id: str) -> bool:
        """Check if repository has base model files for quantization"""
        onnx_path = os.path.join(repo_path, "onnx")
        
        if not os.path.exists(onnx_path):
            return False
        
        # Find all base model files (without mode suffixes)
        base_models = self._find_base_model_files(onnx_path)
        
        if not base_models:
            self.logger.debug(f"No base model files found in {repo_id}")
            return False
        
        self.logger.debug(f"Repository {repo_id} has {len(base_models)} base model files: {base_models}")
        return True
    
    def apply_migration(self, repo_path: str, repo_id: str, interactive: bool = True) -> MigrationResult:
        """Apply model quantization - only add missing variants"""
        try:
            onnx_path = os.path.join(repo_path, "onnx")
            
            # Step 1: Validate existing ONNX models
            if not self._validate_onnx_models(onnx_path, repo_id):
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.FAILED,
                    changes_made=False,
                    error_message="ONNX model validation failed"
                )
            
            
            # Step 2: Get all base model files first
            base_models = self._find_base_model_files(onnx_path)
            
            if not base_models:
                self.logger.info(f"No base model files found in {repo_id}")
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.COMPLETED,
                    changes_made=False,
                    error_message="No base model files found for quantization"
                )
            
            # Step 3: Process each base model file separately with its own temporary directory
            all_quantization_results = {
                "success": False,
                "successful_modes": [],
                "failed_modes": [],
                "failure_logs": {}
            }
            all_modified_files = []
            
            for model_file in base_models:
                # Get detailed information about missing and invalid modes for this specific model
                mode_details = self._get_detailed_missing_quantization_modes(onnx_path, model_file)
                missing_modes = mode_details["missing_modes"] + mode_details["invalid_modes"]
                
                if not missing_modes:
                    self.logger.info(f"Model {model_file} already has all quantized variants")
                    continue
                
                # Track this model as being processed
                self.quantization_details["models_processed"].append(model_file)
                
                self.logger.info(f"Model {model_file} missing modes: {mode_details['missing_modes']}, invalid modes: {mode_details['invalid_modes']}")
                
                # Store the original invalid modes to track which ones get removed later
                original_invalid_modes = set(mode_details["invalid_modes"].copy())
                
                # Create separate temporary directory for this model
                with tempfile.TemporaryDirectory(prefix=f"model_quantization_{model_file[:-5]}_") as temp_dir:
                    input_dir = os.path.join(temp_dir, "input")
                    output_dir = os.path.join(temp_dir, "output")
                    os.makedirs(input_dir, exist_ok=True)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    original_path = os.path.join(onnx_path, model_file)
                    slimmed_path = os.path.join(input_dir, model_file)
                    
                    # Slim the model before quantization
                    if not self._slim_model(original_path, slimmed_path, repo_id):
                        self.logger.error(f"Failed to slim {model_file}, skipping quantization for this model")
                        continue
                    
                    self.logger.info(f"Slimmed and prepared model: {model_file}")
                    
                    # Quantize this specific model with its missing modes
                    quantization_result = self._quantize_models(input_dir, output_dir, repo_id, missing_modes)
                    
                    # Process successful quantized variants
                    if quantization_result["success"] and os.path.exists(output_dir):
                        # Step 4: Copy quantized variants to onnx directory for this model
                        existing_files = set(os.listdir(onnx_path))
                        model_modified_files = []
                        validation_failed_models = []
                        
                        for item in os.listdir(output_dir):
                            if item.endswith('.onnx'):
                                src_path = os.path.join(output_dir, item)
                                dst_path = os.path.join(onnx_path, item)
                                
                                # Extract mode from filename for tracking
                                base_name = model_file[:-5] if model_file.endswith('.onnx') else model_file
                                for mode in self.QUANTIZATION_MODES:
                                    if item == f"{base_name}_{mode}.onnx":
                                        current_mode = mode
                                        break
                                else:
                                    continue  # Skip files that don't match expected pattern
                                
                                # Validate the new model before copying - catch validation errors
                                try:
                                    if self._validate_single_model(src_path, item):
                                        # Check if we're replacing an existing (invalid) file or adding a new one
                                        if item in existing_files:
                                            self.logger.info(f"Replacing invalid quantized model: {item}")
                                            # Track as added for invalid (successful replacement)
                                            if current_mode not in self.quantization_details["added_for_invalid"]:
                                                self.quantization_details["added_for_invalid"].append(f"{model_file}:{current_mode}")
                                        else:
                                            self.logger.info(f"Adding new quantized model: {item}")
                                            # Track as added for missing
                                            if current_mode not in self.quantization_details["added_for_missing"]:
                                                self.quantization_details["added_for_missing"].append(f"{model_file}:{current_mode}")
                                        
                                        shutil.copy2(src_path, dst_path)
                                        model_modified_files.append(f"onnx/{item}")
                                        
                                        # Remove from original_invalid_modes since we successfully replaced it
                                        original_invalid_modes.discard(current_mode)
                                    else:
                                        self.logger.warning(f"Post-conversion validation failed for quantized model: {item}")
                                        validation_failed_models.append(item)
                                except Exception as e:
                                    # Catch any validation errors and continue with other models
                                    error_msg = f"Validation error for {item}: {str(e)}"
                                    self.logger.warning(f"Post-conversion validation error for quantized model: {item} - {error_msg}")
                                    validation_failed_models.append(item)
                        
                        # Add this model's results to the overall results
                        all_modified_files.extend(model_modified_files)
                        
                        # Track any remaining invalid modes that couldn't be regenerated (need to be removed)
                        for invalid_mode in original_invalid_modes:
                            invalid_file = f"{model_file[:-5]}_{invalid_mode}.onnx"
                            invalid_path = os.path.join(onnx_path, invalid_file)
                            if os.path.exists(invalid_path):
                                os.remove(invalid_path)
                                self.quantization_details["removed_invalid"].append(f"{model_file}:{invalid_mode}")
                                self.logger.warning(f"Removed invalid quantized model that couldn't be regenerated: {invalid_file}")
                        
                        # Log validation results for this model
                        if validation_failed_models:
                            self.logger.warning(f"Model {model_file}: Post-conversion validation failed for {len(validation_failed_models)} variants: {validation_failed_models}")
                        if model_modified_files:
                            self.logger.info(f"Model {model_file}: Post-conversion validation passed for {len(model_modified_files)} variants")
                    
                    # Aggregate quantization results
                    if quantization_result["success"]:
                        all_quantization_results["success"] = True
                    all_quantization_results["successful_modes"].extend(quantization_result["successful_modes"])
                    all_quantization_results["failed_modes"].extend(quantization_result["failed_modes"])
                    all_quantization_results["failure_logs"].update(quantization_result["failure_logs"])
                
            # Use aggregated results for final processing
            quantization_result = all_quantization_results
            
            # Log quantization results summary
            if quantization_result["successful_modes"]:
                self.logger.info(f"Successfully quantized modes: {quantization_result['successful_modes']}")
            if quantization_result["failed_modes"]:
                self.logger.warning(f"Failed quantization modes: {quantization_result['failed_modes']}")
                for mode, error in quantization_result["failure_logs"].items():
                    self.logger.debug(f"Failure details for {mode}: {error}")
            
            # Fail only if no modes succeeded at all
            if not quantization_result["success"]:
                error_msg = f"All quantization modes failed. Failed modes: {quantization_result['failed_modes']}"
                if quantization_result["failure_logs"]:
                    # Include first error as example
                    first_error = list(quantization_result["failure_logs"].values())[0]
                    error_msg += f"\nExample error: {first_error[:200]}..."
                
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.FAILED,
                    changes_made=False,
                    error_message=error_msg
                )
            
            # Process final results using all_modified_files
            if all_modified_files:
                self.logger.info(f"Successfully added {len(all_modified_files)} quantized models for {repo_id}")
                
                # Include information about any failed quantization modes
                error_messages = []
                if quantization_result["failed_modes"]:
                    error_messages.append(f"Quantization failed for modes: {', '.join(quantization_result['failed_modes'])}")
                
                error_message = "; ".join(error_messages) if error_messages else None
                
                return MigrationResult(
                    migration_type=self.migration_type,
                    status=MigrationStatus.COMPLETED,
                    changes_made=True,
                    files_modified=all_modified_files,
                    error_message=error_message
                )
            else:
                # No models were successfully added
                if quantization_result["successful_modes"]:
                    # Quantization succeeded but all models failed validation
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.FAILED,
                        changes_made=False,
                        error_message="All generated quantized models failed post-conversion validation"
                    )
                else:
                    # No new models were needed
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.COMPLETED,
                        changes_made=False,
                        error_message="No new quantized models were needed"
                    )
                    
        except Exception as e:
            error_msg = f"Error in model quantization for {repo_id}: {e}"
            self.logger.error(error_msg)
            return MigrationResult(
                migration_type=self.migration_type,
                status=MigrationStatus.FAILED,
                changes_made=False,
                error_message=error_msg
            )
    
    def get_pr_description(self) -> str:
        """Get the PR description with detailed quantization results"""
        
        # Build detailed quantization summary
        summary_parts = []
        
        if self.quantization_details["models_processed"]:
            summary_parts.append(f"**Models processed:** {', '.join(self.quantization_details['models_processed'])}")
        
        if self.quantization_details["added_for_missing"]:
            missing_list = []
            for entry in self.quantization_details["added_for_missing"]:
                model, mode = entry.split(':', 1)
                missing_list.append(f"  - `{model}`: {mode}")
            summary_parts.append(f"**✅ Added quantization modes (missing files):**\n" + "\n".join(missing_list))
        
        if self.quantization_details["added_for_invalid"]:
            invalid_list = []
            for entry in self.quantization_details["added_for_invalid"]:
                model, mode = entry.split(':', 1)
                invalid_list.append(f"  - `{model}`: {mode}")
            summary_parts.append(f"**🔄 Added quantization modes (replaced invalid files):**\n" + "\n".join(invalid_list))
        
        if self.quantization_details["removed_invalid"]:
            removed_list = []
            for entry in self.quantization_details["removed_invalid"]:
                model, mode = entry.split(':', 1)
                removed_list.append(f"  - `{model}`: {mode}")
            summary_parts.append(f"**❌ Removed invalid quantization modes (regeneration failed):**\n" + "\n".join(removed_list))
        
        # Create the full PR description
        description = f"""This automated migration updates the model repository for Transformers.js v3.

## 🔧 Model Binary Migration Results

{chr(10).join(summary_parts) if summary_parts else "No quantization changes were needed."}

---
🤖 Generated by transformers-js-v3-migrator"""

        return description
    
    def _find_base_model_files(self, onnx_path: str) -> list:
        """Find all base model files (without mode suffixes) in the onnx directory"""
        if not os.path.exists(onnx_path):
            return []
        
        existing_files = set(os.listdir(onnx_path))
        base_models = []
        
        # Look for .onnx files that don't have quantization mode suffixes
        for filename in existing_files:
            if filename.endswith('.onnx'):
                # Check if this is a base model (not a quantized variant)
                base_name = filename[:-5]  # Remove .onnx extension
                
                # Skip if this appears to be a quantized variant
                is_quantized_variant = any(
                    base_name.endswith(f"_{mode}") for mode in self.QUANTIZATION_MODES
                )
                
                # Skip if this has "_quantized" suffix
                is_legacy_quantized = base_name.endswith("_quantized")
                
                if not is_quantized_variant and not is_legacy_quantized:
                    base_models.append(filename)
        
        return sorted(base_models)
    
    def _get_missing_quantization_modes(self, onnx_path: str, base_model: str) -> tuple:
        """Get quantization modes that don't already exist or are invalid for a specific base model"""
        existing_files = set(os.listdir(onnx_path))
        missing_modes = []
        
        # Extract base name without .onnx extension
        base_name = base_model[:-5] if base_model.endswith('.onnx') else base_model
        
        for mode in self.QUANTIZATION_MODES:
            variant_file = f"{base_name}_{mode}.onnx"
            
            if variant_file not in existing_files:
                # File doesn't exist - need to generate
                missing_modes.append(mode)
                self.logger.debug(f"Missing quantized model file: {variant_file}")
            else:
                # File exists - validate it
                variant_path = os.path.join(onnx_path, variant_file)
                if not self._validate_single_model_isolated(variant_path, variant_file):
                    # File exists but is invalid - need to regenerate
                    missing_modes.append(mode)
                    self.logger.warning(f"Invalid quantized model file found: {variant_file} - will regenerate")
                else:
                    self.logger.debug(f"Valid quantized model file found: {variant_file}")
        
        return tuple(missing_modes)
    
    def _get_detailed_missing_quantization_modes(self, onnx_path: str, base_model: str) -> dict:
        """Get detailed information about missing and invalid quantization modes"""
        existing_files = set(os.listdir(onnx_path))
        result = {
            "missing_modes": [],      # Modes where file doesn't exist
            "invalid_modes": [],      # Modes where file exists but is invalid
            "valid_modes": []         # Modes where file exists and is valid
        }
        
        # Extract base name without .onnx extension
        base_name = base_model[:-5] if base_model.endswith('.onnx') else base_model
        
        for mode in self.QUANTIZATION_MODES:
            variant_file = f"{base_name}_{mode}.onnx"
            
            if variant_file not in existing_files:
                # File doesn't exist - need to generate
                result["missing_modes"].append(mode)
                self.logger.debug(f"Missing quantized model file: {variant_file}")
            else:
                # File exists - validate it
                variant_path = os.path.join(onnx_path, variant_file)
                if not self._validate_single_model_isolated(variant_path, variant_file):
                    # File exists but is invalid - need to regenerate
                    result["invalid_modes"].append(mode)
                    self.logger.warning(f"Invalid quantized model file found: {variant_file} - will regenerate")
                else:
                    result["valid_modes"].append(mode)
                    self.logger.debug(f"Valid quantized model file found: {variant_file}")
        
        return result
    
    def _validate_onnx_models(self, onnx_path: str, repo_id: str) -> bool:
        """Validate only the base ONNX model files using isolated dependencies via uv"""
        base_models = self._find_base_model_files(onnx_path)
        return self._validate_onnx_models_isolated(onnx_path, repo_id, base_models)
    
    def _validate_onnx_models_isolated(self, onnx_path: str, repo_id: str, base_models: list) -> bool:
        """Validate specific ONNX model files using isolated dependencies via uv"""
        try:
            requirements_file = self.transformers_js_path / "scripts" / "requirements.txt"
            if not requirements_file.exists():
                self.logger.error(f"Requirements file not found at {requirements_file}")
                return False
            
            if not base_models:
                self.logger.info(f"No base models to validate for {repo_id}")
                return True
            
            # Create a validation script that only validates specific files
            model_files_str = "', '".join(base_models)
            validation_script = f'''
import onnx
import os
import sys

onnx_path = "{onnx_path}"
model_files = ['{model_files_str}']

for filename in model_files:
    model_path = os.path.join(onnx_path, filename)
    if os.path.exists(model_path):
        print(f"Validating base ONNX model: {{filename}}")
        onnx.checker.check_model(model_path, full_check=True)
        print(f"✓ Base model {{filename}} is valid")
    else:
        print(f"Warning: Model file {{filename}} not found")
'''
            
            # Run validation using uv with isolated dependencies
            result = subprocess.run([
                "uv", "run", 
                "--with-requirements", str(requirements_file),
                "python", "-c", validation_script
            ], capture_output=True, text=True, check=True)
            
            self.logger.info(f"✓ Base model ONNX validation completed for {len(base_models)} files via isolated environment")
            if result.stdout:
                # Log the validation output
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.info(line)
            return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Isolated ONNX validation failed for {repo_id}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error in isolated ONNX validation for {repo_id}: {e}")
            return False

    def _validate_single_model(self, model_path: str, model_name: str) -> bool:
        """Validate a single ONNX model using isolated dependencies via uv"""
        return self._validate_single_model_isolated(model_path, model_name)
    
    def _slim_model(self, input_path: str, output_path: str, repo_id: str) -> bool:
        """Slim ONNX model using onnxslim with isolated dependencies via uv"""
        try:
            requirements_file = self.transformers_js_path / "scripts" / "requirements.txt"
            if not requirements_file.exists():
                self.logger.error(f"Requirements file not found at {requirements_file}")
                return False
            
            self.logger.info(f"Slimming model: {os.path.basename(input_path)}")
            
            # Run onnxslim using uv with isolated dependencies
            result = subprocess.run([
                "uv", "run", 
                "--with-requirements", str(requirements_file),
                "onnxslim", input_path, output_path
            ], capture_output=True, text=True, check=True)
            
            self.logger.info(f"✓ Successfully slimmed {os.path.basename(input_path)}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"onnxslim failed for {input_path}: {e.stderr}")
            if e.stdout:
                self.logger.error(f"onnxslim stdout: {e.stdout}")
            return False
        except Exception as e:
            self.logger.error(f"Error slimming model {input_path}: {e}")
            return False
    
    def _validate_single_model_isolated(self, model_path: str, model_name: str) -> bool:
        """Validate a single ONNX model using isolated dependencies via uv"""
        try:
            requirements_file = self.transformers_js_path / "scripts" / "requirements.txt"
            if not requirements_file.exists():
                self.logger.warning(f"Requirements file not found, skipping validation of {model_name}")
                return True
            
            validation_script = f'''
import onnx
print("Validating quantized model: {model_name}")
onnx.checker.check_model("{model_path}", full_check=True)
print("✓ Model {model_name} is valid")
'''
            
            result = subprocess.run([
                "uv", "run", 
                "--with-requirements", str(requirements_file),
                "python", "-c", validation_script
            ], capture_output=True, text=True, check=True)
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.info(line)
            return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Isolated validation failed for {model_name}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error in isolated validation for {model_name}: {e}")
            return False
    
    def _quantize_models(self, input_folder: str, output_folder: str, repo_id: str, modes: tuple) -> dict:
        """Quantize models using transformers.js quantization script from submodule
        
        Returns:
            dict: {
                "success": bool,  # True if at least one mode succeeded
                "successful_modes": list,  # List of modes that succeeded
                "failed_modes": list,  # List of modes that failed
                "failure_logs": dict,  # mode -> error message
            }
        """
        result_summary = {
            "success": False,
            "successful_modes": [],
            "failed_modes": [],
            "failure_logs": {}
        }
        
        try:
            if not modes:
                self.logger.info("No quantization modes needed, skipping quantization")
                result_summary["success"] = True
                return result_summary
                
            self.logger.info(f"Quantizing models from {input_folder} with modes: {modes}")
            
            # Verify the transformers.js submodule exists
            if not self.transformers_js_path.exists():
                self.logger.error("transformers.js submodule not found. Please run 'git submodule update --init'")
                return result_summary
            
            # Verify requirements file exists
            requirements_file = self.transformers_js_path / "scripts" / "requirements.txt"
            if not requirements_file.exists():
                self.logger.error(f"Requirements file not found at {requirements_file}")
                return result_summary
            
            # Process each quantization mode individually to handle failures gracefully
            for mode in modes:
                try:
                    self.logger.info(f"Quantizing with mode: {mode}")
                    
                    # Build command for single mode
                    cmd = [
                        "uv", "run", 
                        "--with-requirements", str(requirements_file),
                        "python", "-m", "scripts.quantize",
                        "--input_folder", input_folder,
                        "--output_folder", output_folder,
                        "--modes", mode
                    ]
                    
                    result = subprocess.run(cmd, cwd=str(self.transformers_js_path), capture_output=True, text=True, check=True)
                    
                    self.logger.info(f"✓ Successfully quantized with mode: {mode}")
                    result_summary["successful_modes"].append(mode)
                    
                except subprocess.CalledProcessError as e:
                    error_msg = f"Quantization failed for mode {mode}: {e.stderr}"
                    if e.stdout:
                        error_msg += f"\nStdout: {e.stdout}"
                    
                    self.logger.warning(f"⚠ Quantization mode {mode} failed (this may be expected for some models): {e.stderr}")
                    result_summary["failed_modes"].append(mode)
                    result_summary["failure_logs"][mode] = error_msg
                    
                    # Special handling for q4f16 which is known to fail sometimes
                    if mode == "q4f16":
                        self.logger.info("q4f16 quantization failure is expected for some models, continuing with other modes")
                    
                except Exception as e:
                    error_msg = f"Unexpected error during {mode} quantization: {str(e)}"
                    self.logger.warning(f"⚠ Quantization mode {mode} failed: {error_msg}")
                    result_summary["failed_modes"].append(mode)
                    result_summary["failure_logs"][mode] = error_msg
            
            # Consider successful if at least one mode succeeded
            if result_summary["successful_modes"]:
                result_summary["success"] = True
                self.logger.info(f"✓ Quantization completed for {repo_id}. Successful modes: {result_summary['successful_modes']}")
                if result_summary["failed_modes"]:
                    self.logger.info(f"Failed modes (expected for some models): {result_summary['failed_modes']}")
            else:
                self.logger.error(f"✗ All quantization modes failed for {repo_id}")
            
            return result_summary
            
        except Exception as e:
            error_msg = f"Error in quantization process for {repo_id}: {e}"
            self.logger.error(error_msg)
            result_summary["failure_logs"]["general"] = error_msg
            return result_summary
    
    
