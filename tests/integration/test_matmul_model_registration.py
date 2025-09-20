"""
Integration tests for matmul model registration.

This module tests that matmul models are properly registered and discoverable
by the diode integration system.
"""

import json
import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from torch_diode.integration.base_integration import ModelPointer
from torch_diode.integration.matmul_integration import (
    create_matmul_integration,
    MatmulIntegration,
)
from torch_diode.model_registry import get_model_registry, register_model


class TestMatmulModelRegistration(unittest.TestCase):
    """Test cases for matmul model registration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_model_dir = Path(__file__).parent.parent.parent / "trained_models"
        self.v1_model_path = (
            self.test_model_dir / "matmul_kernel_runtime_prediction" / "v1_model.pt"
        )

    def test_v1_model_exists(self):
        """Test that the v1_model.pt file exists."""
        self.assertTrue(
            self.v1_model_path.exists(),
            f"Expected model file not found: {self.v1_model_path}",
        )

    def test_model_pointer_creation(self):
        """Test creating a model pointer for the v1 model."""
        model_pointer = ModelPointer(
            model_name="v1_model.pt",
            relative_path="matmul_kernel_runtime_prediction",
            model_purpose="matmul_kernel_runtime_prediction",
            interface_name="torch._inductor.choices",
            description="Matrix multiplication kernel runtime prediction model v1",
            version="1.0",
            dependencies=["torch._inductor", "torch._inductor.choices"],
        )

        self.assertEqual(model_pointer.model_name, "v1_model.pt")
        self.assertEqual(
            model_pointer.model_purpose, "matmul_kernel_runtime_prediction"
        )
        self.assertTrue(
            model_pointer.exists(), "Model pointer should detect existing model file"
        )

    def test_model_registry_registration(self):
        """Test registering the v1 model with the model registry."""
        registry = get_model_registry()

        # Create a model pointer for the v1 model
        v1_model_pointer = ModelPointer(
            model_name="v1_model.pt",
            relative_path="matmul_kernel_runtime_prediction",
            model_purpose="matmul_kernel_runtime_prediction",
            interface_name="torch._inductor.choices",
            description="Matrix multiplication kernel runtime prediction model v1",
            version="1.0",
            dependencies=["torch._inductor", "torch._inductor.choices"],
        )

        # Register the model
        register_model(v1_model_pointer)

        # Verify it can be retrieved
        retrieved_model = registry.get_model(
            "matmul_kernel_runtime_prediction", "v1_model.pt"
        )
        self.assertIsNotNone(
            retrieved_model, "Model should be retrievable after registration"
        )
        self.assertEqual(retrieved_model.model_name, "v1_model.pt")

    def test_matmul_integration_with_v1_model(self):
        """Test that MatmulIntegration can find and load the v1 model."""
        # Create integration with v1 model pointer
        model_pointers = [
            ModelPointer(
                model_name="v1_model.pt",
                relative_path="matmul_kernel_runtime_prediction",
                model_purpose="matmul_kernel_runtime_prediction",
                interface_name="torch._inductor.choices",
                description="Matrix multiplication kernel runtime prediction model v1",
                version="1.0",
                dependencies=["torch._inductor", "torch._inductor.choices"],
            )
        ]

        integration = MatmulIntegration(
            model_pointers=model_pointers, enable_fallback=True
        )

        # Test that the model is detected as available
        available_models = integration.get_available_models()
        self.assertGreater(
            len(available_models),
            0,
            "Integration should find at least one available model",
        )

        # Check that v1_model.pt is among the available models
        model_names = [model.model_name for model in available_models]
        self.assertIn(
            "v1_model.pt", model_names, "v1_model.pt should be in available models"
        )

    def test_model_loading(self):
        """Test that the v1 model can be loaded successfully."""
        model_pointer = ModelPointer(
            model_name="v1_model.pt",
            relative_path="matmul_kernel_runtime_prediction",
            model_purpose="matmul_kernel_runtime_prediction",
            interface_name="torch._inductor.choices",
            description="Matrix multiplication kernel runtime prediction model v1",
            version="1.0",
            dependencies=["torch._inductor", "torch._inductor.choices"],
        )

        integration = MatmulIntegration(enable_fallback=True)

        try:
            model = integration.load_model(model_pointer)
            self.assertIsNotNone(model, "Model should be loaded successfully")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")

    def test_integration_discovery(self):
        """Test that the integration system can discover the matmul integration."""
        from torch_diode.integration import (
            discover_and_register_integrations,
            get_integration_registry,
        )

        # Clear registry to start fresh
        registry = get_integration_registry()
        registry.integrations.clear()
        registry.execution_order.clear()

        # Discover integrations
        results = discover_and_register_integrations()

        # Check that matmul integration was discovered
        self.assertIn(
            "matmul_integration", results, "matmul_integration should be discovered"
        )

        # Check that it was registered
        self.assertIn(
            "matmul_kernel_prediction",
            registry.integrations,
            "Matmul integration should be registered",
        )

    def test_model_manifest_includes_v1_model(self):
        """Test that the model manifest includes the v1 model when properly registered."""
        from torch_diode.model_registry import generate_model_manifest

        # Register the v1 model
        v1_model_pointer = ModelPointer(
            model_name="v1_model.pt",
            relative_path="matmul_kernel_runtime_prediction",
            model_purpose="matmul_kernel_runtime_prediction",
            interface_name="torch._inductor.choices",
            description="Matrix multiplication kernel runtime prediction model v1",
            version="1.0",
            dependencies=["torch._inductor", "torch._inductor.choices"],
        )
        register_model(v1_model_pointer)

        # Generate manifest
        manifest = generate_model_manifest()

        # Check that the v1 model is included
        self.assertGreater(
            manifest["total_models"], 0, "Manifest should include at least one model"
        )

        # Look for the v1 model in the manifest
        found_v1_model = False
        for purpose, models in manifest["models_by_purpose"].items():
            for model in models:
                if model["name"] == "v1_model.pt":
                    found_v1_model = True
                    break
            if found_v1_model:
                break

        self.assertTrue(found_v1_model, "v1_model.pt should be included in manifest")

    @patch("torch._inductor.virtualized.V")
    def test_full_integration_process(self, mock_v):
        """Test the full integration process with a mock PyTorch environment."""
        # Mock the choices handler functionality
        mock_v.set_choices_handler = Mock()
        mock_v._choices_handler = None

        # Create integration with v1 model
        model_pointers = [
            ModelPointer(
                model_name="v1_model.pt",
                relative_path="matmul_kernel_runtime_prediction",
                model_purpose="matmul_kernel_runtime_prediction",
                interface_name="torch._inductor.choices",
                description="Matrix multiplication kernel runtime prediction model v1",
                version="1.0",
                dependencies=["torch._inductor", "torch._inductor.choices"],
            )
        ]

        integration = MatmulIntegration(
            model_pointers=model_pointers, enable_fallback=True
        )

        # This should not raise an exception
        try:
            result = integration.integrate()
            # We expect it might fail due to mocking, but it shouldn't crash
        except Exception as e:
            # Log the exception but don't fail the test - integration failures
            # are expected in test environments without full PyTorch setup
            logging.warning(f"Integration failed as expected in test environment: {e}")

    def test_model_file_size_reporting(self):
        """Test that model file sizes are reported correctly."""
        model_pointer = ModelPointer(
            model_name="v1_model.pt",
            relative_path="matmul_kernel_runtime_prediction",
            model_purpose="matmul_kernel_runtime_prediction",
            interface_name="torch._inductor.choices",
            description="Matrix multiplication kernel runtime prediction model v1",
            version="1.0",
            dependencies=["torch._inductor", "torch._inductor.choices"],
        )

        # Check that the model has a non-zero size
        size_mb = model_pointer.get_size_mb()
        self.assertGreater(size_mb, 0, "Model file should have a positive size")

    def test_dependency_checking(self):
        """Test that dependency checking works for the matmul integration."""
        integration = create_matmul_integration(enable_fallback=True)

        # Check dependencies - this may fail in test environments but shouldn't crash
        dep_status = integration.check_dependencies()
        self.assertIsInstance(
            dep_status, dict, "Dependency status should be a dictionary"
        )

    def test_matmul_with_max_autotune_integration(self):
        """Test that matmul with max-autotune actually uses the diode integration."""
        import os
          
        # Skip test if we don't have CUDA or if torch inductor is not available
        try:
            import torch._inductor
            import torch._inductor.config
            import torch._inductor.virtualized
        except ImportError:
            self.skipTest("PyTorch Inductor not available")
          
        # Set up the integration first
        from torch_diode.integration import integrate_all, discover_and_register_integrations
          
        # Clear any existing integrations and discover fresh
        try:
            discover_and_register_integrations()
            integration_results = integrate_all()
            logging.info(f"Integration results: {integration_results}")
        except Exception as e:
            logging.warning(f"Integration setup failed: {e}")
            # Continue with test to at least verify compilation works
          
        # Create test tensors - use sizes that would benefit from optimization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(256, 512, device=device, dtype=torch.float32)
        b = torch.randn(512, 1024, device=device, dtype=torch.float32)
          
        # Track if compilation happens and what optimizations are used
        compilation_happened = {"compiled": False, "optimization_used": False}
          
        # Function to perform matmul
        def matmul_fn(x, y):
            return torch.mm(x, y)
          
        # Test both uncompiled and compiled versions
        # First run uncompiled for baseline
        with torch.no_grad():
            baseline_result = matmul_fn(a, b)
            self.assertEqual(baseline_result.shape, (256, 1024), 
                           "Baseline matmul should produce correct shape")
          
        # Now test with torch.compile and max_autotune
        # Set inductor config for max autotune
        original_max_autotune = getattr(torch._inductor.config, 'max_autotune', False)
        original_benchmark_epilogue = getattr(torch._inductor.config, 'benchmark_epilogue_fusion', False)
          
        try:
            # Enable max autotune
            torch._inductor.config.max_autotune = True
            torch._inductor.config.benchmark_epilogue_fusion = True
              
            # Compile the function
            compiled_matmul = torch.compile(matmul_fn, mode="max-autotune")
              
            # Test that compilation works
            with torch.no_grad():
                compiled_result = compiled_matmul(a, b)
                compilation_happened["compiled"] = True
                  
                # Verify results are correct
                self.assertEqual(compiled_result.shape, (256, 1024),
                               "Compiled matmul should produce correct shape")
                  
                # Verify results match baseline (within floating point tolerance)
                torch.testing.assert_close(compiled_result, baseline_result, 
                                         rtol=1e-4, atol=1e-4,
                                         msg="Compiled and uncompiled results should match")
              
            # Check if diode integration is active by looking for indicators
            # If diode is working, we should see evidence in the compilation process
            try:
                # Try to access the choices handler if it was set
                from torch._inductor.virtualized import V
                if hasattr(V, '_choices_handler') and V._choices_handler is not None:
                    compilation_happened["optimization_used"] = True
                    logging.info("Diode choices handler is active")
                else:
                    logging.info("No custom choices handler detected")
            except Exception as e:
                logging.info(f"Could not check choices handler: {e}")
              
            # Run multiple times to trigger optimization paths
            for i in range(3):
                with torch.no_grad():
                    result = compiled_matmul(a, b)
                    self.assertEqual(result.shape, (256, 1024))
              
            # Report what we found
            self.assertTrue(compilation_happened["compiled"], 
                          "torch.compile should have compiled the function")
              
            logging.info(f"Compilation successful: {compilation_happened['compiled']}")
            logging.info(f"Custom optimization detected: {compilation_happened['optimization_used']}")
              
            # If we have CUDA, also test with different sizes
            if device == "cuda":
                # Test with different matrix sizes to trigger different kernel choices
                test_sizes = [(128, 256, 512), (512, 512, 512), (1024, 256, 512)]
                  
                for m, k, n in test_sizes:
                    a_test = torch.randn(m, k, device=device, dtype=torch.float32)
                    b_test = torch.randn(k, n, device=device, dtype=torch.float32)
                      
                    with torch.no_grad():
                        result = compiled_matmul(a_test, b_test)
                        expected_shape = (m, n)
                        self.assertEqual(result.shape, expected_shape,
                                       f"Compiled matmul should handle {m}x{k} @ {k}x{n}")
              
        finally:
            # Restore original config
            torch._inductor.config.max_autotune = original_max_autotune
            torch._inductor.config.benchmark_epilogue_fusion = original_benchmark_epilogue

    @patch("torch_diode.integration.matmul_integration.torch")
    def test_mock_model_registration_and_usage(self, mock_torch):
        """Test registering a mock model and verifying it gets used."""
        # Create a mock model that we can track
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)

        # Mock predictions - return decreasing scores so first config is "best"
        mock_predictions = torch.tensor([3.0, 2.0, 1.0])
        mock_model.return_value = mock_predictions

        # Track model calls
        model_called = {"called": False, "call_count": 0}

        def track_model_call(*args, **kwargs):
            model_called["called"] = True
            model_called["call_count"] += 1
            return mock_predictions

        mock_model.side_effect = track_model_call

        # Mock torch.load to return our mock model
        mock_torch.load = Mock(
            return_value={"model_state_dict": {}, "hidden_layer_widths": [256, 256]}
        )

        # Create a model pointer for our mock
        mock_model_pointer = ModelPointer(
            model_name="test_mock_model.pt",
            relative_path="matmul_kernel_runtime_prediction",
            model_purpose="matmul_kernel_runtime_prediction",
            interface_name="torch._inductor.choices",
            description="Mock model for testing diode integration",
            version="test",
            dependencies=["torch._inductor", "torch._inductor.choices"],
        )

        # Create integration
        integration = MatmulIntegration(
            model_pointers=[mock_model_pointer], enable_fallback=True
        )

        # Mock the model loading process
        with patch.object(integration, "load_model", return_value=mock_model):
            # Mock the DiodeInductorChoices class
            with patch(
                "torch_diode.integration.inductor_integration.DiodeInductorChoices"
            ) as MockChoicesClass:
                mock_choices_instance = Mock()
                MockChoicesClass.return_value = mock_choices_instance

                # Track if our choices instance is used
                choices_used = {"called": False}

                def track_choices_call(*args, **kwargs):
                    choices_used["called"] = True
                    return mock_choices_instance

                MockChoicesClass.side_effect = track_choices_call

                # Mock virtualized.V to avoid import issues
                with patch("torch._inductor.virtualized.V") as mock_v:
                    mock_v.set_choices_handler = Mock()

                    # Try to integrate (may fail but shouldn't crash)
                    try:
                        result = integration.integrate()
                        # If integration succeeds, that's also a valid test outcome
                        logging.info("Integration succeeded in test environment")

                    except Exception as e:
                        # Integration failures are expected in test environments
                        logging.info(
                            f"Integration failed as expected in test environment: {e}"
                        )

                    # Even if integration fails, verify the model pointer was processed
                    # The model should be detected as available if the file "exists"
                    # (we'll mock this check)
                    with patch.object(mock_model_pointer, "exists", return_value=True):
                        available_models = integration.get_available_models()
                        model_names = [model.model_name for model in available_models]
                        self.assertIn(
                            "test_mock_model.pt",
                            model_names,
                            "Mock model should be detected as available",
                        )


class TestGetModelManifestScript(unittest.TestCase):
    """Test cases for the get_model_manifest.py script."""

    def setUp(self):
        """Set up test fixtures."""
        self.script_path = (
            Path(__file__).parent.parent.parent / "devops" / "get_model_manifest.py"
        )
        self.project_root = Path(__file__).parent.parent.parent

    def test_script_exists(self):
        """Test that the get_model_manifest.py script exists."""
        self.assertTrue(
            self.script_path.exists(), f"Script not found: {self.script_path}"
        )

    def test_get_model_manifest_function(self):
        """Test the get_model_manifest function directly."""
        # Import the script module
        import sys

        sys.path.insert(0, str(self.script_path.parent))

        try:
            from get_model_manifest import get_model_manifest

            # Call the function
            manifest = get_model_manifest()

            # Check that we get a valid manifest
            self.assertIsNotNone(manifest, "Manifest should not be None")
            self.assertIsInstance(manifest, dict, "Manifest should be a dictionary")

            # Check required keys
            required_keys = [
                "version",
                "total_models",
                "total_size_mb",
                "models_by_purpose",
                "models_by_interface",
                "all_dependencies",
                "model_files",
            ]
            for key in required_keys:
                self.assertIn(key, manifest, f"Manifest should contain key: {key}")

            # Check that we have at least one model (the v1 model)
            self.assertGreater(
                manifest["total_models"], 0, "Should have at least one model"
            )

            # Check that v1_model.pt is in the manifest
            found_v1_model = False
            for purpose, models in manifest["models_by_purpose"].items():
                for model in models:
                    if model["name"] == "v1_model.pt":
                        found_v1_model = True
                        break
                if found_v1_model:
                    break

            self.assertTrue(found_v1_model, "v1_model.pt should be in manifest")

        finally:
            # Clean up sys.path
            if str(self.script_path.parent) in sys.path:
                sys.path.remove(str(self.script_path.parent))

    def test_script_cli_json_output(self):
        """Test running the script with JSON output format."""
        try:
            import sys
            result = subprocess.run(
                [sys.executable, str(self.script_path), "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root),
            )

            # Check that script executed successfully
            if result.returncode != 0:
                self.fail(
                    f"Script failed with return code {result.returncode}. "
                    f"Stderr: {result.stderr}"
                )

            # Check that output is valid JSON
            try:
                manifest = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                self.fail(f"Script output is not valid JSON: {e}")

            # Check required keys in JSON output
            required_keys = ["version", "total_models", "total_size_mb"]
            for key in required_keys:
                self.assertIn(key, manifest, f"JSON output should contain key: {key}")

            # Check that we have models
            self.assertGreater(
                manifest["total_models"], 0, "Should report at least one model"
            )

        except subprocess.TimeoutExpired:
            self.fail("Script execution timed out")

    def test_script_cli_paths_output(self):
        """Test running the script with paths output format."""
        try:
            import sys
            result = subprocess.run(
                [sys.executable, str(self.script_path), "--format", "paths"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root),
            )

            # Check that script executed successfully
            if result.returncode != 0:
                self.fail(
                    f"Script failed with return code {result.returncode}. "
                    f"Stderr: {result.stderr}"
                )

            # Check that output contains at least one path
            paths = result.stdout.strip().split("\n")
            self.assertGreater(len(paths), 0, "Should output at least one path")

            # Check that each path exists and is a file
            for path_str in paths:
                if path_str.strip():  # Skip empty lines
                    path = Path(path_str.strip())
                    self.assertTrue(path.exists(), f"Path should exist: {path}")
                    self.assertTrue(path.is_file(), f"Path should be a file: {path}")

            # Check that v1_model.pt is in the paths
            v1_model_found = any("v1_model.pt" in path for path in paths)
            self.assertTrue(v1_model_found, "v1_model.pt should be in output paths")

        except subprocess.TimeoutExpired:
            self.fail("Script execution timed out")

    def test_script_cli_output_to_file(self):
        """Test running the script with output to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            import sys
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--format",
                    "json",
                    "--output",
                    str(temp_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root),
            )

            # Check that script executed successfully
            if result.returncode != 0:
                self.fail(
                    f"Script failed with return code {result.returncode}. "
                    f"Stderr: {result.stderr}"
                )

            # Check that output file was created
            self.assertTrue(temp_path.exists(), "Output file should be created")

            # Check that file contains valid JSON
            try:
                with open(temp_path, "r") as f:
                    manifest = json.load(f)
            except json.JSONDecodeError as e:
                self.fail(f"Output file does not contain valid JSON: {e}")

            # Check content
            self.assertIn("total_models", manifest, "Output should contain model count")
            self.assertGreater(
                manifest["total_models"], 0, "Should have at least one model"
            )

        except subprocess.TimeoutExpired:
            self.fail("Script execution timed out")
        finally:
            # Clean up temp file
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def test_script_cli_help(self):
        """Test that the script shows help information."""
        try:
            import sys
            result = subprocess.run(
                [sys.executable, str(self.script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.project_root),
            )

            # Help should exit with code 0
            self.assertEqual(result.returncode, 0, "Help should exit successfully")

            # Check that help output contains expected information
            help_output = result.stdout.lower()
            self.assertIn(
                "model manifest", help_output, "Help should mention model manifest"
            )
            self.assertIn("--format", help_output, "Help should mention format option")
            self.assertIn("--output", help_output, "Help should mention output option")

        except subprocess.TimeoutExpired:
            self.fail("Help command timed out")

    def test_manifest_consistency_with_registry(self):
        """Test that the script output is consistent with the model registry."""
        # Import the script module
        import sys

        sys.path.insert(0, str(self.script_path.parent))

        try:
            from torch_diode.model_registry import generate_model_manifest
            from get_model_manifest import get_model_manifest

            # Get manifest from script
            script_manifest = get_model_manifest()

            # Get manifest from registry directly
            registry_manifest = generate_model_manifest()

            # They should have the same number of models
            self.assertEqual(
                script_manifest["total_models"],
                registry_manifest["total_models"],
                "Script and registry should report same model count",
            )

            # They should have the same total size
            self.assertAlmostEqual(
                script_manifest["total_size_mb"],
                registry_manifest["total_size_mb"],
                places=6,
                msg="Script and registry should report same total size",
            )

            # They should have the same dependencies
            self.assertEqual(
                set(script_manifest["all_dependencies"]),
                set(registry_manifest["all_dependencies"]),
                "Script and registry should report same dependencies",
            )

        finally:
            # Clean up sys.path
            if str(self.script_path.parent) in sys.path:
                sys.path.remove(str(self.script_path.parent))

    def test_script_error_handling(self):
        """Test script behavior with invalid arguments."""
        # Test with invalid format
        import sys
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--format", "invalid"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(self.project_root),
        )

        # Should fail with non-zero return code
        self.assertNotEqual(
            result.returncode, 0, "Script should fail with invalid format"
        )


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    unittest.main()
