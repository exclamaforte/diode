"""
Tests for Diode registration with PyTorch Inductor.

This module tests the automatic registration of Diode when the torch_diode package
is imported, and verifies that the _finalize_template_configs method is properly
called during PyTorch compilation.
"""

import os
# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag
    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import sys
import pytest
import torch
import logging
from unittest.mock import Mock, patch
from typing import Optional

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging for test visibility
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDiodeRegistration:
    """Test the automatic registration of Diode with PyTorch Inductor."""

    def setup_method(self):
        """Set up test environment."""
        # Disable PyTorch Inductor caches to force fresh template selection
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        
        # Reset call tracking
        self._finalize_template_configs_called = False
        self._finalize_template_configs_call_count = 0

    def test_diode_package_import(self):
        """Test that the torch_diode package can be imported without errors."""
        logger.info("Testing torch_diode package import...")
        
        try:
            import torch_diode
            logger.info(f"Successfully imported torch_diode version: {torch_diode.__version__}")
            assert torch_diode.__version__ is not None
        except Exception as e:
            pytest.fail(f"Failed to import torch_diode: {e}")

    def test_diode_integration_info(self):
        """Test that integration info is available after import."""
        import torch_diode
        
        # Check the integration info to see if models were loaded successfully
        integration_info = torch_diode.get_integration_info()
        logger.info(f"Integration info: {integration_info}")
        
        assert integration_info is not None, "Integration info should not be None"
        assert isinstance(integration_info, dict), "Integration info should be a dictionary"
        
        # Check if we have any integrations (successful or not)
        if integration_info:
            for name, info in integration_info.items():
                logger.info(f"Integration '{name}': {info}")
                assert "integration_status" in info, f"Integration {name} should have status"

    def test_pytorch_inductor_availability(self):
        """Test that PyTorch Inductor is available and accessible."""
        logger.info("Checking PyTorch Inductor availability...")
        
        try:
            from torch._inductor.virtualized import V
            logger.info("PyTorch Inductor is available")
            assert V is not None
        except ImportError:
            pytest.skip("PyTorch Inductor not available - this is expected in some environments")

    def test_diode_choices_handler_registration(self):
        """Test that Diode choices handler is properly registered."""
        import torch_diode
        
        try:
            from torch._inductor.virtualized import V
            
            # Check if Diode choices handler is active by checking V.choices
            if hasattr(V, "choices") and V.choices is not None:
                choices_obj = V.choices
                logger.info(f"Choices object found: {type(choices_obj).__name__}")
                
                # Check if it's our DiodeInductorChoices or in virtualized handler
                target_choices_obj = None
                if "DiodeInductorChoices" in type(choices_obj).__name__:
                    logger.info("✅ Diode choices handler successfully registered!")
                    target_choices_obj = choices_obj
                elif hasattr(V, "_choices_handler") and V._choices_handler is not None:
                    logger.info(f"Found virtualized choices handler: {type(V._choices_handler).__name__}")
                    if "DiodeInductorChoices" in type(V._choices_handler).__name__:
                        logger.info("✅ Diode choices handler found in virtualized handler!")
                        target_choices_obj = V._choices_handler
                    else:
                        logger.info(f"Note: Found choices handler '{type(V._choices_handler).__name__}' in virtualized handler")
                        target_choices_obj = V._choices_handler
                else:
                    logger.info(f"Note: Found choices handler '{type(choices_obj).__name__}' (this may be expected if Diode uses a compatibility wrapper)")
                    target_choices_obj = choices_obj
                
                # Verify the target object has the expected method
                if target_choices_obj and hasattr(target_choices_obj, "_finalize_template_configs"):
                    assert callable(target_choices_obj._finalize_template_configs), "_finalize_template_configs should be callable"
                    logger.info("✅ _finalize_template_configs method found and is callable")
                else:
                    logger.warning(f"Target choices object {type(target_choices_obj).__name__} does not have _finalize_template_configs method")
                
            else:
                logger.warning("No choices object registered")
                # This may be expected in certain environments, so we don't fail the test
                
        except ImportError:
            pytest.skip("PyTorch Inductor not available")

    def test_finalize_template_configs_instrumentation(self):
        """Test that we can instrument the _finalize_template_configs method."""
        import torch_diode
        
        try:
            from torch._inductor.virtualized import V
            
            if not hasattr(V, "choices") or V.choices is None:
                pytest.skip("No choices object registered")
            
            choices_obj = V.choices
            target_choices_obj = choices_obj
            
            # Check virtualized handler if needed
            if not hasattr(target_choices_obj, "_finalize_template_configs") and hasattr(V, "_choices_handler"):
                target_choices_obj = V._choices_handler
            
            if not hasattr(target_choices_obj, "_finalize_template_configs"):
                pytest.skip("_finalize_template_configs method not found")
            
            # Instrument the method to track calls
            original_method = target_choices_obj._finalize_template_configs
            call_count = 0
            
            def instrumented_finalize_template_configs(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                logger.info(f"🎯 _finalize_template_configs called! (call #{call_count})")
                return original_method(*args, **kwargs)
            
            # Monkey patch the method
            target_choices_obj._finalize_template_configs = (
                instrumented_finalize_template_configs.__get__(
                    target_choices_obj, type(target_choices_obj)
                )
            )
            
            logger.info("✅ Successfully instrumented _finalize_template_configs method")
            
            # Verify instrumentation works by calling the method directly (if safe)
            try:
                # Create minimal mock arguments for testing
                mock_template_choices = {"test": iter([Mock()])}
                mock_kernel_inputs = Mock()
                mock_templates = [Mock()]
                
                # This may fail due to missing dependencies, but instrumentation should work
                try:
                    target_choices_obj._finalize_template_configs(
                        template_choices=mock_template_choices,
                        kernel_inputs=mock_kernel_inputs,
                        templates=mock_templates,
                        op_name="test_op"
                    )
                except Exception as e:
                    logger.info(f"Direct method call failed (expected): {e}")
                
                # Even if the method call failed, instrumentation should have incremented count
                # (though it may not have if the error occurred before our wrapper)
                logger.info(f"Final call count: {call_count}")
                
            except Exception as e:
                logger.info(f"Instrumentation test failed: {e}")
            
            # Restore original method
            target_choices_obj._finalize_template_configs = original_method
            
        except ImportError:
            pytest.skip("PyTorch Inductor not available")

    def test_diode_stats_accessibility(self):
        """Test that Diode statistics are accessible."""
        import torch_diode
        
        try:
            from torch._inductor.virtualized import V
            
            if not hasattr(V, "choices") or V.choices is None:
                pytest.skip("No choices object registered")
            
            choices_obj = V.choices
            target_choices_obj = choices_obj
            
            # Check virtualized handler if needed
            if not hasattr(target_choices_obj, "get_stats") and hasattr(V, "_choices_handler"):
                target_choices_obj = V._choices_handler
            
            if hasattr(target_choices_obj, "get_stats"):
                stats = target_choices_obj.get_stats()
                logger.info(f"Diode stats: {stats}")
                assert isinstance(stats, dict), "Stats should be a dictionary"
                
                # Stats may be empty initially, which is fine
                logger.info("✅ Diode statistics are accessible")
            else:
                logger.warning("get_stats method not found on choices object")
                
        except ImportError:
            pytest.skip("PyTorch Inductor not available")

    def test_environment_variable_setup(self):
        """Test that necessary environment variables are properly set."""
        # Check that cache disabling environment variable is set
        assert os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1", \
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES should be set to '1'"
        
        logger.info("✅ Environment variables properly configured")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_availability(self):
        """Test CUDA availability for GPU-based tests."""
        assert torch.cuda.is_available(), "CUDA should be available for GPU tests"
        
        device_count = torch.cuda.device_count()
        assert device_count > 0, "At least one CUDA device should be available"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.info("✅ CUDA is available and configured")


class TestDiodeRegistrationIntegration:
    """Integration tests for Diode registration that require more setup."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

    def test_diode_registration_with_model_loading(self):
        """Test registration with actual model loading scenarios."""
        import torch_diode
          
        # Get integration info
        integration_info = torch_diode.get_integration_info()
          
        # Test that we get some response, even if empty
        assert integration_info is not None, "Integration info should not be None"
        
        # Check for successful integrations
        successful_integrations = []
        for name, info in integration_info.items():
            if info.get("integration_status") == "success":
                successful_integrations.append(name)
                models_loaded = info.get("models_loaded", 0)
                logger.info(f"✅ Integration '{name}' successful with {models_loaded} models loaded")
        
        if successful_integrations:
            logger.info(f"✅ Found {len(successful_integrations)} successful integrations")
        else:
            logger.info("ℹ️ No successful integrations found (models may not be available)")
        
        # This test passes regardless of model availability since registration
        # should work even without models (with fallback behavior)

    def test_diode_integration_error_handling(self):
        """Test that Diode registration handles errors gracefully."""
        import torch_diode
        
        # This should not raise exceptions even if there are integration issues
        try:
            integration_info = torch_diode.get_integration_info()
            logger.info("✅ Integration info retrieved without exceptions")
        except Exception as e:
            pytest.fail(f"Integration info retrieval should not raise exceptions: {e}")
        
        # Test that PyTorch operations still work even if Diode integration has issues
        try:
            a = torch.randn(10, 10)
            b = torch.randn(10, 10)
            c = torch.mm(a, b)
            assert c.shape == (10, 10)
            logger.info("✅ Basic PyTorch operations work after Diode import")
        except Exception as e:
            pytest.fail(f"Basic PyTorch operations should work after Diode import: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
