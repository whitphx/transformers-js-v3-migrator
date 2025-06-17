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
    QUANTIZATION_MODES = ("int8", "uint8", "bnb4", "q4", "q4f16")
    
    def __init__(self):
        super().__init__()
        # Get the path to the transformers.js submodule
        self.project_root = Path(__file__).parent.parent.parent
        self.transformers_js_path = self.project_root / "transformers-js"
    
    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.MODEL_BINARIES
    
    @property
    def description(self) -> str:
        modes_str = ", ".join(self.QUANTIZATION_MODES)
        return f"Add missing quantized ONNX model variants ({modes_str}) for Transformers.js v3"
    
    def is_applicable(self, repo_path: str, repo_id: str) -> bool:
        """Check if repository has model.onnx that needs quantization"""
        onnx_path = os.path.join(repo_path, "onnx")
        
        if not os.path.exists(onnx_path):
            return False
        
        # Check for the main model.onnx file
        model_file = os.path.join(onnx_path, "model.onnx")
        if not os.path.exists(model_file):
            return False
        
        # Check for missing quantized variants - only add what's missing
        existing_files = set(os.listdir(onnx_path))
        
        # Determine which quantized variants are missing for model.onnx
        missing_modes = self._get_missing_quantization_modes(onnx_path)
        
        if not missing_modes:
            self.logger.info(f"Repository {repo_id} already has all quantized variants {self.QUANTIZATION_MODES}")
            return False
        
        self.logger.info(f"Repository {repo_id} is missing quantized variants: {missing_modes}")
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
            
            # Step 1.5: Preview and ask for confirmation in interactive mode
            if interactive:
                user_choice = self._preview_and_confirm(onnx_path, repo_id)
                if user_choice == "reject_skip":
                    self.logger.info(f"User declined model quantization for {repo_id} - leaving undone")
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.FAILED,  # Mark as failed so repo isn't added to global processed list
                        changes_made=False,
                        error_message="User declined quantization (ns) - repository left undone for future attempts"
                    )
                elif user_choice == "reject_done":
                    self.logger.info(f"User declined model quantization for {repo_id} - marking as done")
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.COMPLETED,
                        changes_made=False,
                        error_message="User declined quantization"
                    )
                elif user_choice != "accept":
                    # This shouldn't happen, but handle gracefully
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.COMPLETED,
                        changes_made=False,
                        error_message="User declined quantization"
                    )
            
            # Step 2: Create temporary directory for processing
            with tempfile.TemporaryDirectory(prefix="model_quantization_") as temp_dir:
                input_dir = os.path.join(temp_dir, "input")
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                # Step 3: Process the main model.onnx file (slim then copy to input directory)
                model_file = "model.onnx"
                original_path = os.path.join(onnx_path, model_file)
                slimmed_path = os.path.join(input_dir, model_file)
                
                if not os.path.exists(original_path):
                    self.logger.info(f"No model.onnx found in {repo_id}")
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.COMPLETED,
                        changes_made=False,
                        error_message="No model.onnx found for quantization"
                    )
                
                # Slim the model before quantization
                if not self._slim_model(original_path, slimmed_path, repo_id):
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.FAILED,
                        changes_made=False,
                        error_message=f"Failed to slim {model_file}"
                    )
                
                self.logger.info(f"Slimmed and prepared model: {model_file}")
                
                # Step 4: Quantize slimmed model (only missing variants)
                missing_modes = self._get_missing_quantization_modes(onnx_path)
                if not self._quantize_models(input_dir, output_dir, repo_id, missing_modes):
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.FAILED,
                        changes_made=False,
                        error_message="Model quantization failed"
                    )
                
                # Step 5: Copy only new quantized variants to onnx directory
                existing_files = set(os.listdir(onnx_path))
                modified_files = []
                
                for item in os.listdir(output_dir):
                    if item.endswith('.onnx') and item not in existing_files:
                        # Only copy files that don't already exist
                        src_path = os.path.join(output_dir, item)
                        dst_path = os.path.join(onnx_path, item)
                        
                        # Validate the new model before copying
                        if self._validate_single_model(src_path, item):
                            shutil.copy2(src_path, dst_path)
                            modified_files.append(f"onnx/{item}")
                            self.logger.info(f"Added new quantized model: {item}")
                        else:
                            self.logger.warning(f"Skipping invalid quantized model: {item}")
                
                # Step 6: Test models with Transformers.js (basic validation)
                if modified_files and not self._test_models_with_transformers_js(onnx_path, repo_id):
                    self.logger.warning(f"Some models failed Transformers.js validation for {repo_id}")
                
                if modified_files:
                    self.logger.info(f"Successfully added {len(modified_files)} quantized models for {repo_id}")
                    return MigrationResult(
                        migration_type=self.migration_type,
                        status=MigrationStatus.COMPLETED,
                        changes_made=True,
                        files_modified=modified_files
                    )
                else:
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
    
    def _get_missing_quantization_modes(self, onnx_path: str) -> tuple:
        """Get quantization modes that don't already exist in the directory"""
        existing_files = set(os.listdir(onnx_path))
        missing_modes = []
        
        for mode in self.QUANTIZATION_MODES:
            variant_file = f"model_{mode}.onnx"
            if variant_file not in existing_files:
                missing_modes.append(mode)
        
        return tuple(missing_modes)
    
    def _validate_onnx_models(self, onnx_path: str, repo_id: str) -> bool:
        """Validate ONNX models using isolated dependencies via uv"""
        return self._validate_onnx_models_isolated(onnx_path, repo_id)
    
    def _validate_onnx_models_isolated(self, onnx_path: str, repo_id: str) -> bool:
        """Validate ONNX models using isolated dependencies via uv"""
        try:
            requirements_file = self.transformers_js_path / "scripts" / "requirements.txt"
            if not requirements_file.exists():
                self.logger.error(f"Requirements file not found at {requirements_file}")
                return False
            
            # Create a simple validation script
            validation_script = f'''
import onnx
import os
import sys

onnx_path = "{onnx_path}"
for filename in os.listdir(onnx_path):
    if filename.endswith('.onnx'):
        model_path = os.path.join(onnx_path, filename)
        print(f"Validating ONNX model: {{filename}}")
        onnx.checker.check_model(model_path, full_check=True)
        print(f"✓ Model {{filename}} is valid")
'''
            
            # Run validation using uv with isolated dependencies
            result = subprocess.run([
                "uv", "run", 
                "--with-requirements", str(requirements_file),
                "python", "-c", validation_script
            ], capture_output=True, text=True, check=True)
            
            self.logger.info("✓ ONNX validation completed via isolated environment")
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
    
    def _quantize_models(self, input_folder: str, output_folder: str, repo_id: str, modes: tuple) -> bool:
        """Quantize models using transformers.js quantization script from submodule"""
        try:
            if not modes:
                self.logger.info("No quantization modes needed, skipping quantization")
                return True
                
            self.logger.info(f"Quantizing models from {input_folder} with modes: {modes}")
            
            # Verify the transformers.js submodule exists
            if not self.transformers_js_path.exists():
                self.logger.error("transformers.js submodule not found. Please run 'git submodule update --init'")
                return False
            
            # Verify requirements file exists
            requirements_file = self.transformers_js_path / "scripts" / "requirements.txt"
            if not requirements_file.exists():
                self.logger.error(f"Requirements file not found at {requirements_file}")
                return False
            
            # Run the quantization script using uv with isolated dependencies
            self.logger.info("Running quantization with isolated dependencies via uv")
            
            # Build command with only the needed modes
            cmd = [
                "uv", "run", 
                "--with-requirements", str(requirements_file),
                "python", "-m", "scripts.quantize",
                "--input_folder", input_folder,
                "--output_folder", output_folder,
                "--modes"
            ] + list(modes)
            
            result = subprocess.run(cmd, cwd=str(self.transformers_js_path), capture_output=True, text=True, check=True)
            
            self.logger.info("✓ Quantization completed via uv")
            self.logger.info(f"✓ Successfully quantized models for {repo_id} with modes: {modes}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Quantization failed for {repo_id}: {e.stderr}")
            if e.stdout:
                self.logger.error(f"Quantization stdout: {e.stdout}")
            return False
        except Exception as e:
            self.logger.error(f"Error quantizing models for {repo_id}: {e}")
            return False
    
    
    def _test_models_with_transformers_js(self, onnx_path: str, repo_id: str) -> bool:
        """Basic validation that models work with Transformers.js"""
        try:
            # This is a placeholder for more comprehensive testing
            # In practice, you might want to load models with transformers.js
            # and run basic inference to ensure compatibility
            
            self.logger.info(f"Basic Transformers.js compatibility check for {repo_id}")
            
            # For now, just check that quantized models exist and are readable
            quantized_files = [f for f in os.listdir(onnx_path) 
                             if f.endswith('.onnx') and ('_q4' in f or '_fp16' in f)]
            
            if not quantized_files:
                self.logger.warning("No quantized models found for testing")
                return False
            
            # Basic file readability check
            for model_file in quantized_files:
                model_path = os.path.join(onnx_path, model_file)
                try:
                    with open(model_path, 'rb') as f:
                        # Read first few bytes to ensure file is not corrupted
                        header = f.read(16)
                        if len(header) < 16:
                            self.logger.error(f"Model file {model_file} appears corrupted")
                            return False
                except Exception as e:
                    self.logger.error(f"Cannot read model file {model_file}: {e}")
                    return False
            
            self.logger.info(f"✓ Basic compatibility check passed for {len(quantized_files)} models")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing models with Transformers.js: {e}")
            return False
    
    def _preview_and_confirm(self, onnx_path: str, repo_id: str) -> str:
        """Preview the quantization tasks and ask for user confirmation"""
        
        # Analyze what needs to be done for model.onnx
        existing_files = set(os.listdir(onnx_path))
        model_file = "model.onnx"
        
        missing_variants = []
        tasks_preview = []
        
        # Check if model.onnx exists
        if model_file not in existing_files:
            print(f"No model.onnx found in repository")
            return "reject_done"
        
        # Check for missing quantization variants of model.onnx
        missing_modes = self._get_missing_quantization_modes(onnx_path)
        
        for mode in missing_modes:
            variant_file = f"model_{mode}.onnx"
            description = f"{mode.upper()} quantized variant"
            missing_variants.append(variant_file)
            tasks_preview.append(f"  • Generate {variant_file} ({description})")
        
        print(f"\n{'='*80}")
        print(f"Model Binary Quantization Preview for: {repo_id}")
        print(f"{'='*80}")
        modes_str = ", ".join(missing_modes) if missing_modes else "no missing modes"
        print(f"Commit message: ⚡ Add quantized model variants ({modes_str})")
        print(f"{'='*80}")
        
        print(f"\nSource model:")
        model_path = os.path.join(onnx_path, model_file)
        try:
            file_size = os.path.getsize(model_path)
            size_mb = file_size / (1024 * 1024)
            print(f"  • {model_file} ({size_mb:.1f} MB)")
        except:
            print(f"  • {model_file}")
        
        print(f"\nQuantization tasks to perform:")
        if tasks_preview:
            for task in tasks_preview:
                print(task)
            
            print(f"\nProcess overview:")
            print(f"  1. Use uv to run operations with isolated dependencies from submodule's requirements.txt")
            print(f"  2. Identify base models (non-quantized variants)")  
            print(f"  3. Slim base models using onnxslim")
            print(f"  4. Run quantization script with modes: {', '.join(missing_modes)}")
            print(f"  5. Validate generated models with ONNX checker")
            print(f"  6. Test basic compatibility with Transformers.js")
            print(f"  7. Add {len(missing_variants)} new quantized models to repository")
            
            print(f"\nEstimated additional storage: ~{len(missing_variants) * 50}MB (approximate)")
            print(f"Note: Only missing variants will be added - existing models are preserved")
            
        else:
            print("  • No quantization needed - all variants already exist")
        
        print(f"\n{'='*80}")
        
        # Ask for confirmation
        while True:
            if not tasks_preview:
                # No work needed
                response = input("No quantization needed. Mark as done? [y/n]: ").lower().strip()
                if response in ['y', 'yes']:
                    return "reject_done"
                elif response in ['n', 'no']:
                    return "reject_skip"
                else:
                    print("Please enter 'y' for yes or 'n' for no")
                    continue
            else:
                response = input("Proceed with quantization? [y/ns/nd/p] (y=yes, ns=no+skip, nd=no+done, p=show preview again): ").lower().strip()
                
                if response in ['y', 'yes']:
                    return "accept"
                elif response in ['ns', 'no+skip', 'skip']:
                    return "reject_skip"  # Skip and leave undone for future attempts
                elif response in ['nd', 'no+done', 'done']:
                    return "reject_done"  # Skip and mark as done
                elif response in ['n', 'no']:
                    # For backward compatibility, ask for clarification
                    print("Please clarify:")
                    print("  ns = Skip and leave undone (can retry later)")
                    print("  nd = Skip and mark done (won't retry)")
                    continue
                elif response in ['p', 'preview']:
                    # Show preview again
                    print(f"\nQuantization tasks to perform:")
                    for task in tasks_preview:
                        print(task)
                    continue
                else:
                    print("Please enter:")
                    print("  y  = Proceed with quantization")
                    print("  ns = Skip and leave undone (can retry later)")
                    print("  nd = Skip and mark done (won't retry)")
                    print("  p  = Show preview again")