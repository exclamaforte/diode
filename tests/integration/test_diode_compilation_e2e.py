"""
End-to-end compilation tests for Diode integration with PyTorch.

This module tests that Diode's _finalize_template_configs method is properly
called during actual PyTorch compilation and matrix multiplication operations.
"""

import logging

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import os
import sys

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging for test visibility
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global variable to track if _finalize_template_configs was called
_FINALIZE_TEMPLATE_CONFIGS_CALLED = False
_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT = 0


# class TestDiodeCompilationE2E:
#     """End-to-end tests for Diode integration with PyTorch compilation."""

#     def setup_method(self):
#         """Set up test environment."""
#         global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#         # Reset global tracking
#         _FINALIZE_TEMPLATE_CONFIGS_CALLED = False
#         _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT = 0

#         # AGGRESSIVELY disable ALL PyTorch caches to force fresh template selection
#         os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
#         os.environ["TORCHINDUCTOR_DISABLE_CODECACHE"] = "1"
#         os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_cache_disabled"

#         # CRITICAL: Force PyTorch to use Triton instead of cuBLAS
#         os.environ["DISABLE_ATEN_FALLBACK"] = "1"  # Disable ATEN fallback

#         # EXTREMELY AGGRESSIVE cache clearing
#         try:
#             # Clear dynamo caches
#             torch._dynamo.reset()
#             if hasattr(torch._dynamo, "reset_code"):
#                 torch._dynamo.reset_code()

#             # Clear inductor caches
#             if hasattr(torch._inductor, "codecache") and hasattr(
#                 torch._inductor.codecache, "clear_cache"
#             ):
#                 torch._inductor.codecache.clear_cache()

#             # Clear AOT autograd caches
#             if hasattr(torch._functorch, "_aot_autograd"):
#                 if hasattr(torch._functorch._aot_autograd, "clear_cache"):
#                     torch._functorch._aot_autograd.clear_cache()
#                 # Clear the global cache
#                 if hasattr(torch._functorch._aot_autograd, "autograd_cache"):
#                     if hasattr(torch._functorch._aot_autograd.autograd_cache, "clear"):
#                         torch._functorch._aot_autograd.autograd_cache.clear()

#             # Force garbage collection
#             import gc

#             gc.collect()

#             # Clear CUDA cache if available
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         except Exception as e:
#             pass  # Ignore cache clearing errors

#         # CRITICAL: Set up the diode choices handler globally once
#         try:
#             import torch_diode
#             from torch_diode.integration.inductor_integration import (
#                 DiodeInductorChoices,
#                 install_diode_choices,
#             )

#             # Install DiodeInductorChoices globally so it persists during compilation
#             self.global_diode_choices = install_diode_choices(
#                 model_path=None,  # Will use fallback behavior for this test
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#                 top_k_configs=3,
#                 enable_fallback=True,  # Enable fallback so test doesn't crash if model is missing
#             )

#             logger.info(
#                 f"✅ Globally installed DiodeInductorChoices in setup: {self.global_diode_choices}"
#             )

#         except Exception as e:
#             logger.warning(f"Failed to install DiodeInductorChoices in setup: {e}")
#             self.global_diode_choices = None

#     def _instrument_finalize_template_configs(self):
#         """Instrument the _finalize_template_configs method to track calls."""
#         import torch_diode

#         try:
#             # First, ensure DiodeInductorChoices is registered
#             from torch_diode.integration.inductor_integration import (
#                 DiodeInductorChoices,
#                 install_diode_choices,
#             )

#             logger.info(
#                 "📋 Installing DiodeInductorChoices as the active choices handler..."
#             )

#             # CRITICAL: Use the proper install function to set DiodeInductorChoices
#             # This will force PyTorch to use DiodeInductorChoices instead of default InductorChoices
#             diode_choices = install_diode_choices(
#                 model_path=None,  # Will use fallback behavior for this test
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#                 top_k_configs=3,
#                 enable_fallback=True,  # Enable fallback so test doesn't crash if model is missing
#             )

#             logger.info(f"✅ Installed DiodeInductorChoices: {diode_choices}")

#             from torch._inductor.virtualized import V

#             logger.info(f"V.choices: {V.choices}")
#             logger.info(f"V.choices type: {type(V.choices) if V.choices else None}")

#             if not hasattr(V, "choices") or V.choices is None:
#                 logger.error(
#                     "❌ No V.choices available even after installing DiodeInductorChoices"
#                 )
#                 return None, None

#             choices_obj = V.choices
#             target_choices_obj = choices_obj

#             logger.info(f"choices_obj: {choices_obj}")
#             logger.info(f"choices_obj type: {type(choices_obj)}")
#             logger.info(
#                 f"choices_obj hasattr _finalize_template_configs: {hasattr(choices_obj, '_finalize_template_configs')}"
#             )

#             # Check if target_choices_obj is a DiodeInductorChoices instance
#             if isinstance(target_choices_obj, DiodeInductorChoices):
#                 logger.info(
#                     "✅ SUCCESS: Found DiodeInductorChoices instance as active choices!"
#                 )
#             else:
#                 logger.error(
#                     f"❌ target_choices_obj is not DiodeInductorChoices: {type(target_choices_obj)}"
#                 )
#                 logger.error(
#                     "This is the core problem - PyTorch is not using DiodeInductorChoices!"
#                 )
#                 logger.error(
#                     "Install may have failed or PyTorch is overriding our choices handler"
#                 )
#                 return None, None

#             if not hasattr(target_choices_obj, "_finalize_template_configs"):
#                 logger.error(
#                     f"❌ target_choices_obj does not have _finalize_template_configs method"
#                 )
#                 logger.error(
#                     f"Available methods: {[attr for attr in dir(target_choices_obj) if not attr.startswith('__')]}"
#                 )
#                 return None, None

#             # Instrument the method to track calls and returned configs
#             original_method = target_choices_obj._finalize_template_configs
#             logger.info(
#                 f"✅ Found _finalize_template_configs method: {original_method}"
#             )

#             def instrumented_finalize_template_configs(self, *args, **kwargs):
#                 global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#                 # Set global tracking variables
#                 _FINALIZE_TEMPLATE_CONFIGS_CALLED = True
#                 _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT += 1

#                 # Update stats on self (the DiodeInductorChoices instance)
#                 if hasattr(self, "stats"):
#                     self.stats["total_calls"] = self.stats.get("total_calls", 0) + 1

#                 # Also track on target_choices_obj for the test framework
#                 target_choices_obj._finalize_template_configs_called = True
#                 if not hasattr(
#                     target_choices_obj, "_finalize_template_configs_call_count"
#                 ):
#                     target_choices_obj._finalize_template_configs_call_count = 0
#                 target_choices_obj._finalize_template_configs_call_count += 1

#                 logger.info(
#                     f"🎯 _finalize_template_configs called! (call #{target_choices_obj._finalize_template_configs_call_count})"
#                 )
#                 logger.info(f"   Args: {len(args)} positional args")
#                 logger.info(f"   Args types: {[type(arg) for arg in args]}")
#                 for i, arg in enumerate(args):
#                     logger.info(
#                         f"   Arg {i}: type={type(arg)}, value={str(arg)[:100]}..."
#                     )
#                 if len(args) > 3:
#                     logger.info(f"   Op name: {args[3]}")

#                 # Call original method correctly with unpacked arguments
#                 result = original_method(*args, **kwargs)

#                 # Track the number of configs returned
#                 if not hasattr(target_choices_obj, "_returned_config_counts"):
#                     target_choices_obj._returned_config_counts = []
#                 target_choices_obj._returned_config_counts.append(
#                     len(result) if result else 0
#                 )

#                 logger.info(
#                     f"   🔧 Returned {len(result) if result else 0} configurations"
#                 )

#                 return result

#             # Monkey patch the method
#             target_choices_obj._finalize_template_configs = (
#                 instrumented_finalize_template_configs.__get__(
#                     target_choices_obj, type(target_choices_obj)
#                 )
#             )

#             # Set instrumentation markers that we can check later
#             target_choices_obj._instrumentation_successful = True
#             target_choices_obj._finalize_template_configs_called = (
#                 False  # This will be set to True when method is called
#             )
#             target_choices_obj._finalize_template_configs_call_count = 0
#             target_choices_obj._returned_config_counts = []

#             logger.info("✅ Instrumented _finalize_template_configs method")
#             return target_choices_obj, original_method

#         except ImportError:
#             return None, None

#     def _restore_finalize_template_configs(self, target_choices_obj, original_method):
#         """Restore the original _finalize_template_configs method."""
#         if target_choices_obj and original_method:
#             target_choices_obj._finalize_template_configs = original_method
#             logger.info("✅ Restored original _finalize_template_configs method")

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
#     def test_matmul_with_torch_compile(self):
#         """Test matrix multiplication with torch.compile to trigger _finalize_template_configs."""
#         import torch_diode

#         # Instrument the method
#         target_obj, original_method = self._instrument_finalize_template_configs()
#         if not target_obj:
#             pytest.skip(
#                 "_finalize_template_configs method not available for instrumentation"
#             )

#         try:
#             device = "cuda"
#             logger.info(f"Using device: {device}")

#             # Log GPU information
#             gpu_name = torch.cuda.get_device_name(0)
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
#             logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

#             # Use larger dimensions to ensure template selection is triggered
#             M, N, K = 2048, 4096, 1024
#             logger.info(f"Testing mm with dimensions: A({M}x{K}) @ B({K}x{N})")

#             A = torch.randn(M, K, device=device, dtype=torch.float16)
#             B = torch.randn(K, N, device=device, dtype=torch.float16)

#             # Log GPU memory usage
#             allocated_memory = torch.cuda.memory_allocated(0) / 1e6
#             logger.info(f"GPU memory allocated: {allocated_memory:.1f}MB")

#             # Test regular matmul first
#             logger.info("Testing regular matmul...")
#             C_regular = torch.mm(A, B)
#             logger.info(f"Regular matmul result shape: {C_regular.shape}")

#             # Clear compilation caches
#             try:
#                 torch._dynamo.reset()
#                 torch._inductor.codecache.clear_cache()
#                 logger.info("Cleared compilation caches")
#             except Exception as e:
#                 logger.debug(f"Cache clearing failed (not critical): {e}")

#             # Test with torch.compile with aggressive settings
#             logger.info("Testing matmul with torch.compile...")

#             import torch._inductor.config as inductor_config

#             old_debug = getattr(inductor_config, "debug", False)
#             old_max_autotune = getattr(inductor_config, "max_autotune", False)
#             old_max_autotune_gemm = getattr(inductor_config, "max_autotune_gemm", False)

#             try:
#                 # Force aggressive settings to trigger template selection
#                 inductor_config.max_autotune = True
#                 inductor_config.max_autotune_gemm = True
#                 inductor_config.debug = True

#                 from torch._inductor.virtualized import V

#                 # CRITICAL: Install DiodeInductorChoices BEFORE defining the compiled function
#                 # This ensures it's active during compilation/template selection
#                 from torch_diode.integration.inductor_integration import (
#                     DiodeInductorChoices,
#                     install_diode_choices,
#                 )

#                 # Force install DiodeInductorChoices BEFORE compilation
#                 diode_choices = install_diode_choices(
#                     model_path=None,
#                     device="cuda" if torch.cuda.is_available() else "cpu",
#                     top_k_configs=3,
#                     enable_fallback=True,
#                 )

#                 logger.info(f"🎯 BEFORE compilation: V.choices = {V.choices}")
#                 logger.info(f"🎯 V.choices type = {type(V.choices)}")
#                 logger.info(
#                     f"🎯 DiodeInductorChoices installed: {isinstance(V.choices, DiodeInductorChoices)}"
#                 )

#                 @torch.compile(
#                     mode="max-autotune",
#                     backend="inductor",
#                     fullgraph=True,
#                     dynamic=False,
#                 )
#                 def compiled_matmul(a, b):
#                     return torch.mm(a, b)

#                 logger.info(
#                     "Calling compiled matmul (this should trigger _finalize_template_configs)..."
#                 )
#                 call_count_before = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )

#                 C_compiled = compiled_matmul(A, B)

#                 call_count_after = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )
#                 calls_made = call_count_after - call_count_before

#                 logger.info(
#                     f"_finalize_template_configs called {calls_made} times for this operation"
#                 )
#                 logger.info(f"Compiled matmul result shape: {C_compiled.shape}")

#                 # Verify results are close
#                 if torch.allclose(C_regular, C_compiled, rtol=1e-3, atol=1e-3):
#                     logger.info("✅ Regular and compiled matmul results match!")
#                 else:
#                     logger.warning("⚠️  Regular and compiled matmul results differ")

#                 # Check if _finalize_template_configs was called using both global and object tracking
#                 global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#                 finalize_called_global = _FINALIZE_TEMPLATE_CONFIGS_CALLED
#                 finalize_called_object = getattr(
#                     target_obj, "_finalize_template_configs_called", False
#                 )

#                 if finalize_called_global and finalize_called_object:
#                     logger.info(
#                         f"✅ SUCCESS: _finalize_template_configs was called during compilation! (Global: {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT} times, Object tracking: {calls_made} times)"
#                     )

#                     # Verify instrumented method tracked returned configs
#                     returned_config_counts = getattr(
#                         target_obj, "_returned_config_counts", []
#                     )
#                     if returned_config_counts:
#                         logger.info(
#                             f"   📊 Instrumented method tracked {len(returned_config_counts)} calls with config counts: {returned_config_counts}"
#                         )
#                     else:
#                         logger.warning(
#                             "   ⚠️  No returned config counts tracked by instrumented method"
#                         )

#                 else:
#                     logger.warning(
#                         f"⚠️  _finalize_template_configs was not called consistently (Global: {finalize_called_global}, Object: {finalize_called_object})"
#                     )

#             finally:
#                 # Restore original settings
#                 inductor_config.debug = old_debug
#                 inductor_config.max_autotune = old_max_autotune
#                 inductor_config.max_autotune_gemm = old_max_autotune_gemm

#         finally:
#             # Restore original method
#             self._restore_finalize_template_configs(target_obj, original_method)

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
#     def test_multiple_matmul_operations(self):
#         """Test multiple different matmul operations to increase chances of triggering _finalize_template_configs."""
#         import torch_diode

#         # Instrument the method
#         target_obj, original_method = self._instrument_finalize_template_configs()
#         if not target_obj:
#             pytest.skip(
#                 "_finalize_template_configs method not available for instrumentation"
#             )

#         try:
#             device = "cuda"
#             logger.info(f"Using device: {device}")

#             # Test cases with different dimensions
#             test_cases = [
#                 ("mm", 2048, 4096, 1024),
#                 ("mm", 4096, 2048, 512),
#                 ("mm", 1024, 1024, 1024),
#             ]

#             total_calls_before = getattr(
#                 target_obj, "_finalize_template_configs_call_count", 0
#             )

#             for i, (op_type, M, N, K) in enumerate(test_cases):
#                 logger.info(
#                     f"Test case {i+1}: {op_type} with dimensions: A({M}x{K}) @ B({K}x{N})"
#                 )

#                 A = torch.randn(M, K, device=device, dtype=torch.float16)
#                 B = torch.randn(K, N, device=device, dtype=torch.float16)

#                 # Clear caches for each test case
#                 try:
#                     torch._dynamo.reset()
#                     torch._inductor.codecache.clear_cache()
#                 except Exception as e:
#                     logger.debug(f"Cache clearing failed: {e}")

#                 # Configure for aggressive template selection
#                 import torch._inductor.config as inductor_config

#                 old_settings = {
#                     "max_autotune": getattr(inductor_config, "max_autotune", False),
#                     "max_autotune_gemm": getattr(
#                         inductor_config, "max_autotune_gemm", False
#                     ),
#                 }

#                 try:
#                     inductor_config.max_autotune = True
#                     inductor_config.max_autotune_gemm = True

#                     @torch.compile(
#                         mode="max-autotune",
#                         backend="inductor",
#                         fullgraph=True,
#                         dynamic=False,
#                     )
#                     def compiled_matmul(a, b):
#                         return torch.mm(a, b)

#                     call_count_before = getattr(
#                         target_obj, "_finalize_template_configs_call_count", 0
#                     )
#                     C = compiled_matmul(A, B)
#                     call_count_after = getattr(
#                         target_obj, "_finalize_template_configs_call_count", 0
#                     )

#                     calls_for_this_case = call_count_after - call_count_before
#                     logger.info(
#                         f"  -> _finalize_template_configs called {calls_for_this_case} times"
#                     )
#                     logger.info(f"  -> Result shape: {C.shape}")

#                 finally:
#                     # Restore settings
#                     for key, value in old_settings.items():
#                         setattr(inductor_config, key, value)

#             total_calls_after = getattr(
#                 target_obj, "_finalize_template_configs_call_count", 0
#             )
#             total_calls_made = total_calls_after - total_calls_before

#             logger.info(
#                 f"📊 TOTAL: _finalize_template_configs called {total_calls_made} times across all test cases"
#             )

#             # Check if any calls were made using both global and object tracking
#             global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#             finalize_called_global = _FINALIZE_TEMPLATE_CONFIGS_CALLED
#             finalize_called_object = getattr(
#                 target_obj, "_finalize_template_configs_called", False
#             )

#             if (
#                 finalize_called_global
#                 and finalize_called_object
#                 and total_calls_made > 0
#             ):
#                 logger.info(
#                     f"✅ SUCCESS: _finalize_template_configs was called during multiple operations! (Global: {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT} times, Object tracking: {total_calls_made} times)"
#                 )

#                 # Verify instrumented method tracked returned configs
#                 returned_config_counts = getattr(
#                     target_obj, "_returned_config_counts", []
#                 )
#                 if returned_config_counts:
#                     logger.info(
#                         f"   📊 Instrumented method tracked {len(returned_config_counts)} calls with config counts: {returned_config_counts}"
#                     )
#                 else:
#                     logger.warning(
#                         "   ⚠️  No returned config counts tracked by instrumented method"
#                     )

#             else:
#                 logger.warning(
#                     f"⚠️  _finalize_template_configs was not called consistently (Global: {finalize_called_global}, Object: {finalize_called_object}, Total calls: {total_calls_made})"
#                 )

#         finally:
#             # Restore original method
#             self._restore_finalize_template_configs(target_obj, original_method)

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
#     def test_batch_and_addmm_operations(self):
#         """Test batch matmul and addmm operations."""
#         import torch_diode

#         # Instrument the method
#         target_obj, original_method = self._instrument_finalize_template_configs()
#         if not target_obj:
#             pytest.skip(
#                 "_finalize_template_configs method not available for instrumentation"
#             )

#         try:
#             device = "cuda"
#             logger.info(f"Using device: {device}")

#             import torch._inductor.config as inductor_config

#             old_settings = {
#                 "max_autotune": getattr(inductor_config, "max_autotune", False),
#                 "max_autotune_gemm": getattr(
#                     inductor_config, "max_autotune_gemm", False
#                 ),
#             }

#             try:
#                 inductor_config.max_autotune = True
#                 inductor_config.max_autotune_gemm = True

#                 total_calls_before = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )

#                 # Test batch matmul
#                 logger.info("Testing batch matmul...")
#                 batch_size = 4
#                 M, N, K = 256, 512, 128
#                 A_batch = torch.randn(
#                     batch_size, M, K, device=device, dtype=torch.float16
#                 )
#                 B_batch = torch.randn(
#                     batch_size, K, N, device=device, dtype=torch.float16
#                 )

#                 @torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
#                 def compiled_bmm(a, b):
#                     return torch.bmm(a, b)

#                 call_count_before = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )
#                 C_batch = compiled_bmm(A_batch, B_batch)
#                 call_count_after = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )

#                 bmm_calls = call_count_after - call_count_before
#                 logger.info(f"Batch matmul result shape: {C_batch.shape}")
#                 logger.info(
#                     f"_finalize_template_configs called {bmm_calls} times for bmm"
#                 )

#                 # Test addmm operation
#                 logger.info("Testing addmm...")
#                 bias = torch.randn(M, N, device=device, dtype=torch.float16)
#                 A = torch.randn(M, K, device=device, dtype=torch.float16)
#                 B = torch.randn(K, N, device=device, dtype=torch.float16)

#                 @torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
#                 def compiled_addmm(bias, a, b):
#                     return torch.addmm(bias, a, b)

#                 call_count_before = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )
#                 C_addmm = compiled_addmm(bias, A, B)
#                 call_count_after = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )

#                 addmm_calls = call_count_after - call_count_before
#                 logger.info(f"Addmm result shape: {C_addmm.shape}")
#                 logger.info(
#                     f"_finalize_template_configs called {addmm_calls} times for addmm"
#                 )

#                 total_calls_after = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )
#                 total_calls_made = total_calls_after - total_calls_before

#                 logger.info(
#                     f"📊 TOTAL: _finalize_template_configs called {total_calls_made} times for bmm and addmm"
#                 )

#                 # Check if any calls were made using both global and object tracking
#                 global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#                 finalize_called_global = _FINALIZE_TEMPLATE_CONFIGS_CALLED
#                 finalize_called_object = getattr(
#                     target_obj, "_finalize_template_configs_called", False
#                 )

#                 if (
#                     finalize_called_global
#                     and finalize_called_object
#                     and total_calls_made > 0
#                 ):
#                     logger.info(
#                         f"✅ SUCCESS: _finalize_template_configs was called during batch/addmm operations! (Global: {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT} times, Object tracking: {total_calls_made} times)"
#                     )

#                     # Verify instrumented method tracked returned configs
#                     returned_config_counts = getattr(
#                         target_obj, "_returned_config_counts", []
#                     )
#                     if returned_config_counts:
#                         logger.info(
#                             f"   📊 Instrumented method tracked {len(returned_config_counts)} calls with config counts: {returned_config_counts}"
#                         )
#                     else:
#                         logger.warning(
#                             "   ⚠️  No returned config counts tracked by instrumented method"
#                         )

#                 else:
#                     logger.warning(
#                         f"⚠️  _finalize_template_configs was not called consistently (Global: {finalize_called_global}, Object: {finalize_called_object}, Total calls: {total_calls_made})"
#                     )

#             finally:
#                 # Restore settings
#                 for key, value in old_settings.items():
#                     setattr(inductor_config, key, value)

#         finally:
#             # Restore original method
#             self._restore_finalize_template_configs(target_obj, original_method)

#     def test_diode_stats_during_compilation(self):
#         """Test that Diode statistics are updated during compilation."""
#         import torch_diode

#         try:
#             from torch._inductor.virtualized import V

#             if not hasattr(V, "choices") or V.choices is None:
#                 pytest.skip("No choices object registered")

#             choices_obj = V.choices
#             target_choices_obj = choices_obj

#             # Check virtualized handler if needed
#             if not hasattr(target_choices_obj, "get_stats") and hasattr(
#                 V, "_choices_handler"
#             ):
#                 target_choices_obj = V._choices_handler

#             if not hasattr(target_choices_obj, "get_stats"):
#                 pytest.skip("get_stats method not available")

#             # Get initial stats
#             initial_stats = target_choices_obj.get_stats()
#             logger.info(f"Initial stats: {initial_stats}")

#             # The stats should be accessible and be a dictionary
#             assert isinstance(initial_stats, dict), "Stats should be a dictionary"

#             # Note: We don't run actual compilation here to check stats updates
#             # because that would require CUDA and might be flaky. The main
#             # compilation tests above handle the actual stat verification.

#             logger.info("✅ Diode statistics are accessible during test setup")

#         except ImportError:
#             pytest.skip("PyTorch Inductor not available")

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
#     def test_exhaustive_config_filtering_stats(self):
#         """Test that Diode properly filters down from exhaustive configurations with high configs_filtered count."""
#         import torch_diode

#         # Instrument the method
#         target_obj, original_method = self._instrument_finalize_template_configs()
#         if not target_obj:
#             pytest.skip(
#                 "_finalize_template_configs method not available for instrumentation"
#             )

#         try:
#             device = "cuda"
#             logger.info(f"Using device: {device}")

#             # Get initial stats
#             initial_stats = target_obj.get_stats()
#             logger.info(f"Initial stats: {initial_stats}")

#             # Use problem sizes that will generate many exhaustive configs
#             # Large matrices are more likely to have extensive template configurations
#             M, N, K = 4096, 8192, 2048
#             logger.info(
#                 f"Testing exhaustive filtering with large dimensions: A({M}x{K}) @ B({K}x{N})"
#             )

#             A = torch.randn(M, K, device=device, dtype=torch.float16)
#             B = torch.randn(K, N, device=device, dtype=torch.float16)

#             # Clear compilation caches
#             try:
#                 torch._dynamo.reset()
#                 torch._inductor.codecache.clear_cache()
#                 logger.info("Cleared compilation caches")
#             except Exception as e:
#                 logger.debug(f"Cache clearing failed (not critical): {e}")

#             # Configure for maximum template generation
#             import torch._inductor.config as inductor_config

#             old_settings = {
#                 "max_autotune": getattr(inductor_config, "max_autotune", False),
#                 "max_autotune_gemm": getattr(
#                     inductor_config, "max_autotune_gemm", False
#                 ),
#                 "debug": getattr(inductor_config, "debug", False),
#                 "max_autotune_gemm_backends": getattr(
#                     inductor_config, "max_autotune_gemm_backends", None
#                 ),
#                 "force_disable_caches": getattr(
#                     inductor_config, "force_disable_caches", False
#                 ),
#                 "disable_cpp_codegen": getattr(
#                     inductor_config, "disable_cpp_codegen", False
#                 ),
#             }

#             try:
#                 # CRITICAL: Force exhaustive template selection - THIS IS THE KEY!
#                 # These settings are REQUIRED for PyTorch to use Triton templates at all
#                 inductor_config.max_autotune = True
#                 inductor_config.max_autotune_gemm = True
#                 inductor_config.debug = True
#                 inductor_config.force_disable_caches = True
#                 inductor_config.disable_cpp_codegen = True

#                 # Force Triton to be used by disabling other backends
#                 inductor_config.max_autotune_gemm_backends = "TRITON"

#                 # CRITICAL: Disable cuBLAS/ATEN fallback to force template selection
#                 os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
#                 os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#                 # More aggressive forcing to use Triton templates
#                 from torch._inductor import config as inductor_config_module
#                 inductor_config_module.cpp.threads = 1

#                 # Additional settings to ensure templates are used
#                 if hasattr(inductor_config_module, 'max_autotune_allow_flexible_layouts'):
#                     inductor_config_module.max_autotune_allow_flexible_layouts = True

#                 # Make sure we have a reasonable problem size for templates
#                 assert (
#                     M >= 64 and N >= 64 and K >= 64
#                 ), f"Problem size {M}x{N}x{K} might be too small for template selection"

#                 # DEBUG: Check what V.choices is before compilation
#                 from torch._inductor.virtualized import V

#                 logger.info(f"🔍 BEFORE compilation: V.choices = {V.choices}")
#                 logger.info(f"🔍 V.choices type = {type(V.choices)}")
#                 logger.info(
#                     f"🔍 DiodeInductorChoices is active: {type(V.choices).__name__ == 'DiodeInductorChoices'}"
#                 )

#                 # CRITICAL: Re-verify that our DiodeInductorChoices is still the active handler
#                 from torch_diode.integration.inductor_integration import (
#                     DiodeInductorChoices,
#                 )

#                 if not isinstance(V.choices, DiodeInductorChoices):
#                     logger.error(
#                         f"❌ CRITICAL: V.choices is not DiodeInductorChoices! Got {type(V.choices)}"
#                     )
#                     logger.error(
#                         "   This means our DiodeInductorChoices was overridden!"
#                     )
#                     logger.error("   Forcing reinstallation of DiodeInductorChoices...")

#                     # Force reinstall
#                     from torch_diode.integration.inductor_integration import (
#                         install_diode_choices,
#                     )

#                     diode_choices = install_diode_choices(
#                         model_path=None,
#                         device="cuda" if torch.cuda.is_available() else "cpu",
#                         top_k_configs=3,
#                         enable_fallback=True,
#                     )
#                     logger.info(f"🔄 Forced reinstall: V.choices = {V.choices}")

#                     # Make sure target_obj points to the active handler
#                     target_obj = V.choices
#                 else:
#                     logger.info("✅ DiodeInductorChoices is still the active handler")

#                 # Additional debug: check that our instrumentation is still active
#                 if hasattr(target_obj, "_instrumentation_successful"):
#                     logger.info(
#                         f"✅ Instrumentation marker found: {target_obj._instrumentation_successful}"
#                     )
#                 else:
#                     logger.error(
#                         "❌ Instrumentation marker NOT found - method may not be instrumented!"
#                     )

#                 @torch.compile(
#                     mode="max-autotune",
#                     backend="inductor",
#                     fullgraph=True,
#                     dynamic=False,
#                 )
#                 def compiled_matmul_exhaustive(a, b):
#                     return torch.mm(a, b)

#                 logger.info(
#                     "Calling compiled matmul with exhaustive config generation..."
#                 )

#                 # Get stats before compilation
#                 stats_before = target_obj.get_stats()

#                 C_compiled = compiled_matmul_exhaustive(A, B)

#                 # DEBUG: Check what V.choices is after compilation
#                 logger.info(f"🔍 AFTER compilation: V.choices = {V.choices}")
#                 logger.info(f"🔍 V.choices type = {type(V.choices)}")

#                 # Get stats after compilation
#                 stats_after = target_obj.get_stats()

#                 logger.info(f"Compiled matmul result shape: {C_compiled.shape}")

#                 # Calculate the changes in statistics
#                 model_selections = stats_after.get(
#                     "model_selections", 0
#                 ) - stats_before.get("model_selections", 0)
#                 configs_filtered = stats_after.get(
#                     "configs_filtered", 0
#                 ) - stats_before.get("configs_filtered", 0)
#                 total_calls = stats_after.get("total_calls", 0) - stats_before.get(
#                     "total_calls", 0
#                 )
#                 fallback_no_model = stats_after.get(
#                     "fallback_no_model", 0
#                 ) - stats_before.get("fallback_no_model", 0)

#                 logger.info(f"📊 STATISTICS ANALYSIS:")
#                 logger.info(
#                     f"   Total calls to _finalize_template_configs: {total_calls}"
#                 )
#                 logger.info(f"   Model selections (successful): {model_selections}")
#                 logger.info(f"   Configs filtered from exhaustive: {configs_filtered}")
#                 logger.info(f"   Fallbacks due to no model: {fallback_no_model}")

#                 # Additional detailed stats
#                 all_stats = target_obj.get_stats()
#                 logger.info(f"   All statistics: {all_stats}")

#                 # CRITICAL: Check instrumentation success first
#                 instrumentation_successful = getattr(
#                     target_obj, "_instrumentation_successful", False
#                 )

#                 if not instrumentation_successful:
#                     logger.error("❌ CRITICAL: Instrumentation was not successful!")
#                     logger.error(
#                         "   The _finalize_template_configs method was not properly instrumented"
#                     )
#                     assert False, "Instrumentation failed - cannot proceed with test"

#                 logger.info(
#                     "✅ Instrumentation was successful - method was properly replaced"
#                 )

#                 # Check method call tracking
#                 global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#                 finalize_called_global = _FINALIZE_TEMPLATE_CONFIGS_CALLED
#                 finalize_called_object = getattr(
#                     target_obj, "_finalize_template_configs_called", False
#                 )

#                 object_call_count = getattr(
#                     target_obj, "_finalize_template_configs_call_count", 0
#                 )

#                 logger.info(f"📊 INSTRUMENTATION STATUS:")
#                 logger.info(
#                     f"   Instrumentation successful: {instrumentation_successful}"
#                 )
#                 logger.info(
#                     f"   Global tracking: called={finalize_called_global}, count={_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT}"
#                 )
#                 logger.info(
#                     f"   Object tracking: called={finalize_called_object}, count={object_call_count}"
#                 )
#                 logger.info(f"   Stats tracking: {total_calls} calls via stats")

#                 if finalize_called_global and finalize_called_object:
#                     logger.info(
#                         f"✅ SUCCESS: _finalize_template_configs was called {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT} times globally, {object_call_count} times tracked on object!"
#                     )

#                     # Verify instrumented method tracked returned configs
#                     returned_config_counts = getattr(
#                         target_obj, "_returned_config_counts", []
#                     )
#                     if returned_config_counts:
#                         logger.info(
#                             f"   📊 Instrumented method tracked {len(returned_config_counts)} calls with config counts: {returned_config_counts}"
#                         )
#                         logger.info(
#                             f"   📊 Total configs returned across all calls: {sum(returned_config_counts)}"
#                         )
#                     else:
#                         logger.warning(
#                             "   ⚠️  No returned config counts tracked by instrumented method"
#                         )

#                     # Also verify the object stats are consistent
#                     if total_calls > 0:
#                         logger.info(
#                             f"✅ Object stats show {total_calls} calls - consistent with instrumentation tracking!"
#                         )
#                     else:
#                         logger.warning(
#                             f"⚠️  Global shows {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT} calls but object stats show {total_calls}"
#                         )

#                 else:
#                     # This is the core issue - template selection is not being triggered at all
#                     logger.error(
#                         "❌ PROBLEM: _finalize_template_configs was instrumented but never called!"
#                     )
#                     logger.error(
#                         f"   Global tracking: {finalize_called_global} (count: {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT})"
#                     )
#                     logger.error(
#                         f"   Object tracking: {finalize_called_object} (count: {object_call_count})"
#                     )
#                     logger.error(f"   Stats tracking: {total_calls} calls")
#                     logger.error(
#                         "   This means PyTorch is not using the DiodeInductorChoices instance we instrumented."
#                     )
#                     logger.error(
#                         "   Either PyTorch is using cuBLAS or creating a new choices handler during compilation."
#                     )
#                     logger.error(
#                         "   This is why the autotuning over the filtered set isn't happening."
#                     )

#                     # Additional debugging info
#                     logger.error(f"   Current V.choices object: {V.choices}")
#                     logger.error(f"   Current V.choices type: {type(V.choices)}")
#                     logger.error(
#                         f"   Expected DiodeInductorChoices object: {target_obj}"
#                     )
#                     logger.error(
#                         f"   Are they the same object? {V.choices is target_obj}"
#                     )

#                 # Template selection MUST work now - no more skipping!
#                 assert finalize_called_global and finalize_called_object, (
#                     f"❌ FAILURE: _finalize_template_configs was instrumented but never called! "
#                     f"Global: {finalize_called_global} (count: {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT}), "
#                     f"Object: {finalize_called_object} (count: {object_call_count}). "
#                     "Template selection is not working. This means PyTorch is using cuBLAS "
#                     "instead of Triton templates, or creating a new choices handler that bypasses our instrumentation."
#                 )

#                 assert (
#                     _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT > 0
#                 ), f"Expected _finalize_template_configs call count > 0, got {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT}"

#                 logger.info(
#                     f"✅ SUCCESS: _finalize_template_configs instrumentation working correctly!"
#                 )

#                 # Check if we have successful model selections or expected fallbacks
#                 if model_selections > 0:
#                     logger.info("✅ SUCCESS: Model was used for config selection!")

#                     # When model is used, we expect significant filtering from exhaustive configs
#                     # The user specifically wants this to be a high number like 1000
#                     if configs_filtered >= 100:  # Set a reasonable threshold
#                         logger.info(
#                             f"✅ EXCELLENT: High number of configs filtered ({configs_filtered}) - shows exhaustive -> filtered pipeline working!"
#                         )
#                     elif configs_filtered > 0:
#                         logger.info(
#                             f"✅ GOOD: Some configs filtered ({configs_filtered}) - pipeline is working"
#                         )
#                     else:
#                         logger.warning(
#                             f"⚠️  Expected configs_filtered > 0, but got {configs_filtered}"
#                         )

#                     # Additional assertions for successful model usage
#                     assert (
#                         model_selections > 0
#                     ), f"Expected model_selections > 0, got {model_selections}"
#                     # Note: configs_filtered might be 0 if the exhaustive set is already small or if the model selects all configs

#                 elif fallback_no_model > 0:
#                     logger.info(
#                         "ℹ️  Model not available - fallback behavior occurred (this is acceptable)"
#                     )
#                     logger.info(
#                         "   This indicates Diode registration worked but no trained model was available"
#                     )
#                 else:
#                     logger.warning(
#                         "⚠️  Unexpected: No model selections and no fallbacks recorded"
#                     )

#                 # Verify that _finalize_template_configs was actually called during compilation
#                 finalize_called = getattr(
#                     target_obj, "_finalize_template_configs_called", False
#                 )
#                 if finalize_called:
#                     logger.info(
#                         "✅ SUCCESS: _finalize_template_configs was called during compilation!"
#                     )
#                 else:
#                     logger.warning(
#                         "⚠️  _finalize_template_configs was not called (may indicate template selection was bypassed)"
#                     )

#             finally:
#                 # Restore original settings
#                 for key, value in old_settings.items():
#                     setattr(inductor_config, key, value)

#         finally:
#             # Restore original method
#             self._restore_finalize_template_configs(target_obj, original_method)

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
#     def test_returned_config_count_matches_top_k(self):
#         """Test that the number of returned configurations matches the expected top-k value."""
#         import torch_diode

#         # Instrument the method
#         target_obj, original_method = self._instrument_finalize_template_configs()
#         if not target_obj:
#             pytest.skip(
#                 "_finalize_template_configs method not available for instrumentation"
#             )

#         try:
#             device = "cuda"
#             logger.info(f"Using device: {device}")

#             # Check if we have a DiodeInductorChoices instance
#             top_k_expected = getattr(target_obj, "top_k_configs", None)
#             if top_k_expected is None:
#                 logger.info("No Diode model available - will test fallback behavior")
#                 top_k_expected = (
#                     20  # Default PyTorch behavior typically returns ~20 configs
#                 )

#             logger.info(f"Expected top_k configurations: {top_k_expected}")

#             # Use problem sizes that will trigger template selection
#             M, N, K = 2048, 4096, 1024
#             logger.info(f"Testing with dimensions: A({M}x{K}) @ B({K}x{N})")

#             A = torch.randn(M, K, device=device, dtype=torch.float16)
#             B = torch.randn(K, N, device=device, dtype=torch.float16)

#             # Clear compilation caches
#             try:
#                 torch._dynamo.reset()
#                 torch._inductor.codecache.clear_cache()
#                 logger.info("Cleared compilation caches")
#             except Exception as e:
#                 logger.debug(f"Cache clearing failed (not critical): {e}")

#             # Configure for template selection
#             import torch._inductor.config as inductor_config

#             old_settings = {
#                 "max_autotune": getattr(inductor_config, "max_autotune", False),
#                 "max_autotune_gemm": getattr(
#                     inductor_config, "max_autotune_gemm", False
#                 ),
#                 "debug": getattr(inductor_config, "debug", False),
#             }

#             try:
#                 inductor_config.max_autotune = True
#                 inductor_config.max_autotune_gemm = True
#                 inductor_config.debug = True

#                 @torch.compile(
#                     mode="max-autotune",
#                     backend="inductor",
#                     fullgraph=True,
#                     dynamic=False,
#                 )
#                 def compiled_matmul_with_config_check(a, b):
#                     return torch.mm(a, b)

#                 logger.info("Calling compiled matmul to check returned config count...")

#                 # Reset the tracking list
#                 if hasattr(target_obj, "_returned_config_counts"):
#                     target_obj._returned_config_counts = []

#                 C_compiled = compiled_matmul_with_config_check(A, B)
#                 logger.info(f"Compiled matmul result shape: {C_compiled.shape}")

#                 # Check the returned config counts
#                 returned_config_counts = getattr(
#                     target_obj, "_returned_config_counts", []
#                 )
#                 logger.info(f"📊 CONFIG COUNT ANALYSIS:")
#                 logger.info(
#                     f"   Calls to _finalize_template_configs: {len(returned_config_counts)}"
#                 )

#                 if returned_config_counts:
#                     for i, count in enumerate(returned_config_counts):
#                         logger.info(f"   Call #{i+1}: returned {count} configurations")

#                     # Get the configuration counts for verification
#                     max_configs_returned = (
#                         max(returned_config_counts) if returned_config_counts else 0
#                     )
#                     min_configs_returned = (
#                         min(returned_config_counts) if returned_config_counts else 0
#                     )

#                     logger.info(
#                         f"   Max configs returned in any call: {max_configs_returned}"
#                     )
#                     logger.info(
#                         f"   Min configs returned in any call: {min_configs_returned}"
#                     )

#                     # Check if we have a Diode model or fallback behavior
#                     stats = target_obj.get_stats()
#                     model_selections = stats.get("model_selections", 0)
#                     fallback_no_model = stats.get("fallback_no_model", 0)

#                     if model_selections > 0:
#                         # Diode model was used - check against top_k
#                         logger.info(
#                             "✅ Diode model was used for configuration selection"
#                         )

#                         # Assert that returned configs are within reasonable bounds of top_k
#                         for i, count in enumerate(returned_config_counts):
#                             if count > 0:  # Only check non-zero counts
#                                 # SUCCESS CASE: Diode model is working correctly
#                                 # The model should return ≤ top_k configs (or very close)
#                                 if count <= top_k_expected:
#                                     logger.info(
#                                         f"✅ PERFECT: Call #{i+1} returned {count} configs ≤ top_k={top_k_expected}"
#                                     )
#                                     logger.info(
#                                         f"   🎯 This means the Diode model successfully filtered configs!"
#                                     )
#                                 elif count <= top_k_expected * 2:
#                                     logger.info(
#                                         f"✅ GOOD: Call #{i+1} returned {count} configs ≤ {top_k_expected * 2} (within tolerance)"
#                                     )
#                                 else:
#                                     # This would be the original problem the user reported
#                                     logger.error(
#                                         f"❌ PROBLEM: Call #{i+1} returned {count} configs >> top_k={top_k_expected}"
#                                     )
#                                     assert False, (
#                                         f"ORIGINAL ISSUE DETECTED: Call #{i+1} returned {count} configs, "
#                                         f"expected ≤ {top_k_expected}. This means autotuning is running over "
#                                         f"the full exhaustive set instead of the filtered top-k set from Diode."
#                                     )

#                         logger.info(
#                             f"✅ SUCCESS: Diode model is working correctly - configs are properly filtered!"
#                         )

#                     elif fallback_no_model > 0:
#                         # Fallback behavior - expect default PyTorch behavior
#                         logger.info(
#                             "ℹ️  Fallback behavior (no model) - checking against default expectations"
#                         )

#                         # PyTorch default behavior typically returns many configs (10-30)
#                         for i, count in enumerate(returned_config_counts):
#                             if count > 0:
#                                 # In fallback mode, we expect the default PyTorch behavior
#                                 # which usually returns more configs than our top_k
#                                 logger.info(
#                                     f"   Call #{i+1}: {count} configs (fallback behavior)"
#                                 )

#                                 # ISSUE DETECTION: This is the exact problem the user mentioned!
#                                 # The autotuning is running over the full set (e.g., 20) instead of top-k (e.g., 3)
#                                 if (
#                                     count > top_k_expected * 5
#                                 ):  # If way more than expected
#                                     logger.error(
#                                         f"❌ PROBLEM DETECTED: Call #{i+1} returned {count} configs, much higher than top_k={top_k_expected}"
#                                     )
#                                     logger.error(
#                                         f"   This confirms the issue: autotuning is running over the full {count} shapes instead of filtered top-{top_k_expected}"
#                                     )
#                                     logger.error(
#                                         f"   The system should return {top_k_expected} configs for autotuning, not {count}"
#                                     )

#                                     # This is exactly what the user wants us to assert - fail the test when this happens
#                                     assert False, (
#                                         f"AUTOTUNING ISSUE: Expected autotuning over ≤{top_k_expected * 2} configs "
#                                         f"but got {count} configs. This means autotuning is running over the full "
#                                         f"exhaustive set instead of the filtered top-k set from Diode."
#                                     )

#                         logger.info("✅ SUCCESS: Fallback behavior working as expected")

#                     else:
#                         logger.warning(
#                             "⚠️  Unexpected: No model selections and no fallbacks recorded"
#                         )

#                         # Still check that we're not returning an excessive number of configs
#                         for i, count in enumerate(returned_config_counts):
#                             if count > 100:  # Sanity check - should never be this high
#                                 logger.error(
#                                     f"❌ Call #{i+1}: returned {count} configs, this seems excessive"
#                                 )
#             finally:
#                 for key, value in old_settings.items():
#                     setattr(inductor_config, key, value)

#             # Check if _finalize_template_configs was called using both global and object tracking
#             global _FINALIZE_TEMPLATE_CONFIGS_CALLED, _FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT

#             finalize_called_global = _FINALIZE_TEMPLATE_CONFIGS_CALLED
#             finalize_called_object = getattr(
#                 target_obj, "_finalize_template_configs_called", False
#             )

#             if finalize_called_global and finalize_called_object:
#                 logger.info(
#                     f"✅ SUCCESS: _finalize_template_configs instrumentation working correctly! (Global: {_FINALIZE_TEMPLATE_CONFIGS_CALL_COUNT} times, Object tracking confirmed)"
#                 )

#                 # Additional verification that the instrumented method correctly tracked calls
#                 if returned_config_counts:
#                     logger.info(
#                         f"   📊 Instrumented method tracked {len(returned_config_counts)} calls with config counts: {returned_config_counts}"
#                     )
#                 else:
#                     logger.warning(
#                         "   ⚠️  No returned config counts tracked by instrumented method"
#                     )

#             else:
#                 logger.warning(
#                     f"⚠️  _finalize_template_configs instrumentation inconsistent (Global: {finalize_called_global}, Object: {finalize_called_object})"
#                 )

#         finally:
#             # Restore original settings
#             # Restore original method
#             self._restore_finalize_template_configs(target_obj, original_method)

#     def test_environment_setup_for_compilation(self):
#         """Test that the environment is properly set up for compilation tests."""
#         # Check environment variables
#         assert (
#             os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1"
#         ), "TORCHINDUCTOR_FORCE_DISABLE_CACHES should be set"

#         # Check PyTorch availability
#         assert torch.__version__ is not None, "PyTorch should be available"

#         # Check that torch_diode can be imported
#         try:
#             import torch_diode

#             assert torch_diode.__version__ is not None
#         except ImportError:
#             pytest.fail("torch_diode should be importable")

#         logger.info("✅ Environment properly set up for compilation tests")


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
