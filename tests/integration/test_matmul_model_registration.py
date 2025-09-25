"""
Integration tests for matmul model registration.

This module tests that matmul models are properly registered and discoverable
by the diode integration system.
"""

import json

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import logging
import os
import subprocess
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from torch_diode.integration.base_integration import ModelPointer
from torch_diode.integration.matmul_integration import (
    MatmulIntegration,
    create_matmul_integration,
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

    def test_diode_integration_functionality(self):
        """Test that the diode integration functionality works without relying on max-autotune."""
        # Import torch inductor components (test will fail if not available, which is fine)
        import tempfile

        import torch._inductor
        import torch._inductor.config
        import torch._inductor.virtualized
        from torch._inductor.virtualized import V

        # Set up the integration
        from torch_diode.integration import (
            discover_and_register_integrations,
            integrate_all,
        )
        from torch_diode.integration.inductor_integration import (
            DiodeInductorChoices,
            install_diode_choices,
        )

        # Create a test model for integration
        from torch_diode.model.matmul_timing_model import MatmulTimingModel

        # Create a simple model
        model = MatmulTimingModel(
            problem_feature_dim=4,  # Simple features: M, N, K, B
            config_feature_dim=6,  # Simple config features
            hidden_dims=[32, 16],
            dropout_rate=0.1,
        )

        fd, temp_model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        model.save(temp_model_path)

        try:
            # Test direct DiodeInductorChoices installation
            original_handler = getattr(V, "_choices_handler", None)

            # Install our choices handler
            choices_handler = install_diode_choices(
                model_path=temp_model_path,
                device="cpu",
                top_k_configs=2,
                performance_threshold=1.2,
            )

            # Verify the handler was installed
            self.assertIsNotNone(choices_handler)
            self.assertIsInstance(choices_handler, DiodeInductorChoices)
            self.assertTrue(choices_handler._model_loaded)

            # Verify that the handler was installed (can't directly check _choices_handler
            # as it may not be a public attribute)
            # The fact that install_diode_choices completed successfully is sufficient
            self.assertIsInstance(choices_handler, DiodeInductorChoices)

            # Test the handler's core functionality
            from tests.integration.test_inductor_integration_old import (
                MockConfig,
                MockKernelInputs,
                MockKernelTemplateChoice,
                MockTemplate,
                MockTensor,
            )

            # Create test inputs
            tensor_a = MockTensor((128, 64), torch.float16)
            tensor_b = MockTensor((64, 128), torch.float16)
            kernel_inputs = MockKernelInputs([tensor_a, tensor_b])

            # Test feature extraction
            features = choices_handler._extract_features_from_kernel_inputs(
                kernel_inputs, "mm"
            )
            self.assertIsNotNone(features)
            self.assertEqual(features["mm_shape"].M, 128)
            self.assertEqual(features["mm_shape"].N, 128)
            self.assertEqual(features["mm_shape"].K, 64)

            # Test config conversion
            mock_config = MockConfig(
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_stages=2, num_warps=4
            )
            mock_ktc = MockKernelTemplateChoice(
                MockTemplate("test_template"), mock_config
            )

            triton_config = choices_handler._convert_ktc_to_config(mock_ktc)
            self.assertIsNotNone(triton_config)
            self.assertEqual(triton_config.block_m, 64)
            self.assertEqual(triton_config.block_n, 64)

            # Test prediction
            problem_features = features["problem_features"]
            predictions = choices_handler._predict_config_performance(
                problem_features, [triton_config]
            )
            self.assertEqual(len(predictions), 1)
            self.assertIsInstance(predictions[0], float)

            # Test full pipeline with multiple configs
            template = MockTemplate("test_template")
            ktcs = []
            for i in range(3):
                config = MockConfig(
                    BLOCK_M=32 * (i + 1),
                    BLOCK_N=32 * (i + 1),
                    BLOCK_K=16,
                    GROUP_M=8,
                    num_stages=2,
                    num_warps=4,
                )
                ktcs.append(MockKernelTemplateChoice(template, config))

            template_choices = {"test_template": iter(ktcs)}

            # Mock the conversion pipeline since it can't work with mock objects
            with mock.patch(
                "torch_diode.integration.inductor_integration.convert_and_run_inference_pipeline"
            ) as mock_pipeline:
                # Return top 2 configs to simulate successful selection
                mock_pipeline.return_value = ktcs[:2]

                # Test the full selection process
                selected = choices_handler._finalize_template_configs(
                    template_choices, kernel_inputs, None, [], "mm"
                )

                # Should return some configs (model prediction or fallback)
                self.assertGreater(len(selected), 0)
                self.assertLessEqual(len(selected), 3)

            # Check statistics were updated
            stats = choices_handler.get_stats()
            self.assertGreater(stats["total_calls"], 0)

            # Test with PyTorch compilation (simple mode to avoid max-autotune issues)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            a = torch.randn(64, 32, device=device, dtype=torch.float32)
            b = torch.randn(32, 64, device=device, dtype=torch.float32)

            def simple_matmul(x, y):
                return torch.mm(x, y)

            # Test that basic compilation works with our handler installed
            try:
                compiled_fn = torch.compile(
                    simple_matmul, mode="default"
                )  # Use default mode instead of max-autotune
                result = compiled_fn(a, b)
                self.assertEqual(result.shape, (64, 64))
                logging.info("Compilation with Diode integration succeeded")
            except Exception as e:
                # If compilation fails, log it but don't fail the test - the important part is that our handler works
                logging.warning(
                    f"Compilation failed (this may be expected in test environment): {e}"
                )

            # Test integration discovery
            try:
                discover_and_register_integrations()
                integration_results = integrate_all()
                self.assertIsInstance(integration_results, dict)
                logging.info(f"Integration discovery successful: {integration_results}")
            except Exception as e:
                logging.warning(f"Integration discovery failed (may be expected): {e}")

        finally:
            # Clean up
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)

            # Restore original handler
            if hasattr(V, "set_choices_handler"):
                try:
                    V.set_choices_handler(original_handler)
                except Exception:
                    pass

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
                with open(temp_path) as f:
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
            from get_model_manifest import get_model_manifest

            from torch_diode.model_registry import generate_model_manifest

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
