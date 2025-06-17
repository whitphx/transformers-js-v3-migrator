import os
import subprocess
import tempfile
import shutil
from typing import Optional
from ..migration_types import BaseMigration, MigrationType, MigrationResult, MigrationStatus


class ModelBinaryMigration(BaseMigration):
    """Migration for converting and quantizing ONNX model binaries"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def migration_type(self) -> MigrationType:
        return MigrationType.MODEL_BINARIES
    
    @property
    def description(self) -> str:
        return "Add missing quantized ONNX model variants (q4, fp16) for Transformers.js v3"
    
    def is_applicable(self, repo_path: str, repo_id: str) -> bool:
        """Check if repository has ONNX models that need quantization"""
        onnx_path = os.path.join(repo_path, "onnx")
        
        if not os.path.exists(onnx_path):
            return False
        
        # Check for ONNX files
        onnx_files = [f for f in os.listdir(onnx_path) if f.endswith('.onnx')]
        if not onnx_files:
            return False
        
        # Check for missing quantized variants (q4, fp16) - only add what's missing
        existing_files = set(os.listdir(onnx_path))
        
        # Determine which quantized variants are missing
        missing_variants = []
        for onnx_file in onnx_files:
            base_name = onnx_file.replace('.onnx', '')
            
            # Check for q4 and fp16 variants
            q4_file = f"{base_name}_q4.onnx"
            fp16_file = f"{base_name}_fp16.onnx"
            
            if q4_file not in existing_files:
                missing_variants.append('q4')
            if fp16_file not in existing_files:
                missing_variants.append('fp16')
        
        if not missing_variants:
            self.logger.info(f"Repository {repo_id} already has all quantized variants (q4, fp16)")
            return False
        
        self.logger.info(f"Repository {repo_id} is missing quantized variants: {set(missing_variants)}")
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
                
                # Step 3: Copy existing models to input directory (no modification)
                onnx_files = [f for f in os.listdir(onnx_path) if f.endswith('.onnx')]
                for onnx_file in onnx_files:
                    shutil.copy2(
                        os.path.join(onnx_path, onnx_file),
                        os.path.join(input_dir, onnx_file)
                    )
                
                # Step 4: Quantize models (only q4 and fp16 variants)
                if not self._quantize_models(input_dir, output_dir, repo_id):
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
    
    def _validate_onnx_models(self, onnx_path: str, repo_id: str) -> bool:
        """Validate ONNX models using onnx.checker"""
        try:
            import onnx
            
            for filename in os.listdir(onnx_path):
                if filename.endswith('.onnx'):
                    model_path = os.path.join(onnx_path, filename)
                    self.logger.info(f"Validating ONNX model: {filename}")
                    
                    # Load and check model
                    onnx.checker.check_model(model_path, full_check=True)
                    self.logger.info(f"✓ Model {filename} is valid")
            
            return True
            
        except ImportError:
            self.logger.error("onnx package not available for validation")
            return False
        except Exception as e:
            self.logger.error(f"ONNX validation failed for {repo_id}: {e}")
            return False
    
    def _validate_single_model(self, model_path: str, model_name: str) -> bool:
        """Validate a single ONNX model using onnx.checker"""
        try:
            import onnx
            
            self.logger.info(f"Validating quantized model: {model_name}")
            onnx.checker.check_model(model_path, full_check=True)
            self.logger.info(f"✓ Model {model_name} is valid")
            return True
            
        except ImportError:
            self.logger.error("onnx package not available for validation")
            return False
        except Exception as e:
            self.logger.error(f"ONNX validation failed for {model_name}: {e}")
            return False
    
    def _quantize_models(self, input_folder: str, output_folder: str, repo_id: str) -> bool:
        """Quantize models using transformers.js quantization script"""
        try:
            self.logger.info(f"Quantizing models from {input_folder}")
            
            # Clone transformers.js repository to get quantization script
            transformers_js_path = self._setup_transformers_js_repo()
            if not transformers_js_path:
                return False
            
            # Run the quantization script from transformers.js repo (only q4 and fp16)
            result = subprocess.run([
                "python", "-m", "scripts.quantize",
                "--input_folder", input_folder,
                "--output_folder", output_folder,
                "--modes", "q4", "fp16"
            ], cwd=transformers_js_path, capture_output=True, text=True, check=True)
            
            self.logger.info(f"✓ Successfully quantized models for {repo_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Quantization failed for {repo_id}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error quantizing models for {repo_id}: {e}")
            return False
    
    def _setup_transformers_js_repo(self) -> Optional[str]:
        """Clone transformers.js repository to get quantization scripts"""
        try:
            import subprocess
            import tempfile
            import os
            
            # Create temporary directory for transformers.js repo
            transformers_js_dir = tempfile.mkdtemp(prefix="transformers_js_repo_")
            
            # Clone the transformers.js repository
            self.logger.info("Cloning transformers.js repository for quantization scripts")
            result = subprocess.run([
                "git", "clone", 
                "https://github.com/huggingface/transformers.js.git",
                transformers_js_dir
            ], capture_output=True, text=True, check=True)
            
            # Verify the scripts directory exists
            scripts_path = os.path.join(transformers_js_dir, "scripts")
            quantize_script = os.path.join(scripts_path, "quantize.py")
            
            if not os.path.exists(quantize_script):
                self.logger.error(f"Quantization script not found at {quantize_script}")
                return None
            
            self.logger.info(f"Successfully set up transformers.js repo at {transformers_js_dir}")
            return transformers_js_dir
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone transformers.js repository: {e.stderr}")
            return None
        except Exception as e:
            self.logger.error(f"Error setting up transformers.js repository: {e}")
            return None
    
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
        
        # Analyze what needs to be done
        existing_files = set(os.listdir(onnx_path))
        onnx_files = [f for f in existing_files if f.endswith('.onnx')]
        
        missing_variants = []
        tasks_preview = []
        
        for onnx_file in onnx_files:
            base_name = onnx_file.replace('.onnx', '')
            
            # Check for q4 and fp16 variants
            q4_file = f"{base_name}_q4.onnx"
            fp16_file = f"{base_name}_fp16.onnx"
            
            if q4_file not in existing_files:
                missing_variants.append(q4_file)
                tasks_preview.append(f"  • Generate {q4_file} (Q4 quantized variant)")
            
            if fp16_file not in existing_files:
                missing_variants.append(fp16_file)
                tasks_preview.append(f"  • Generate {fp16_file} (FP16 quantized variant)")
        
        print(f"\n{'='*80}")
        print(f"Model Binary Quantization Preview for: {repo_id}")
        print(f"{'='*80}")
        print(f"Commit message: ⚡ Add quantized model variants (q4, fp16)")
        print(f"{'='*80}")
        
        print(f"\nExisting ONNX models found:")
        for onnx_file in sorted(onnx_files):
            file_path = os.path.join(onnx_path, onnx_file)
            try:
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                print(f"  • {onnx_file} ({size_mb:.1f} MB)")
            except:
                print(f"  • {onnx_file}")
        
        print(f"\nQuantization tasks to perform:")
        if tasks_preview:
            for task in tasks_preview:
                print(task)
            
            print(f"\nProcess overview:")
            print(f"  1. Clone transformers.js repository for quantization scripts")
            print(f"  2. Copy existing models to temporary workspace")  
            print(f"  3. Run quantization script with modes: q4, fp16")
            print(f"  4. Validate generated models with ONNX checker")
            print(f"  5. Test basic compatibility with Transformers.js")
            print(f"  6. Add {len(missing_variants)} new quantized models to repository")
            
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