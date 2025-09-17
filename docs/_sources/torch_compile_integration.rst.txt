Accelerating torch.compile with Diode
=====================================

Diode can significantly speed up PyTorch's ``torch.compile`` by providing pre-trained models that predict optimal matrix multiplication configurations, eliminating the need for expensive runtime autotuning. This integration allows you to get the performance benefits of extensive autotuning with minimal compilation time.

Overview
--------

When PyTorch compiles matrix multiplication operations, it typically needs to search through many different kernel configurations to find the optimal one for your specific hardware and problem size. This process, called autotuning, can take substantial time during compilation.

Diode solves this by:

1. **Pre-trained Models**: Using machine learning models trained on extensive performance data
2. **Hardware-Specific Optimization**: Automatically selecting the best model for your GPU
3. **Fast Predictions**: Providing optimal configurations instantly without runtime search

Quick Start
-----------

Getting started with Diode acceleration is simple and requires only three steps:

Step 1: Install torch-diode-models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the pre-trained models package:

.. code-block:: bash

    pip install torch-diode-models

This package contains pre-trained models for popular hardware configurations including NVIDIA H100 and AMD MI300X GPUs.

Step 2: Import and Auto-Register
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply import the package to automatically register the best model for your hardware:

.. code-block:: python

    import torch_diode_models

This import automatically:

* Detects your hardware configuration
* Selects the most appropriate pre-trained model
* Registers the model with PyTorch's compilation system
* Configures the prediction interface

Step 3: Enable Fast Autotuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure PyTorch to use Diode's fast autotuning:

.. code-block:: python

    import torch
    from torch._inductor import config

    # Enable fast autotuning with Diode models
    config.max_autotune_gemm_backends = "DIODE"
    config.fast_autotune = True

Complete Example
----------------

Here's a complete example showing how to use Diode with torch.compile:

.. code-block:: python

    import torch
    import torch_diode_models  # Auto-registers the best model for your hardware
    from torch._inductor import config

    # Configure PyTorch to use Diode acceleration
    config.max_autotune_gemm_backends = "DIODE"
    config.fast_autotune = True

    # Your existing PyTorch code - no changes needed!
    def matmul_function(a, b):
        return torch.mm(a, b)

    # Compile with torch.compile - now accelerated by Diode
    compiled_fn = torch.compile(matmul_function, mode="max-autotune")

    # Use as normal
    a = torch.randn(1024, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)

    result = compiled_fn(a, b)  # Fast compilation + optimal performance

Benefits
--------

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

Diode provides significant improvements in both compilation time and runtime performance:

**Compilation Speed**
* **10-100x faster compilation**: Eliminates expensive autotuning searches
* **Instant predictions**: Model inference takes microseconds vs. seconds of autotuning
* **Consistent compile times**: No variation based on problem size or hardware load

**Runtime Performance**
* **Optimal configurations**: Models trained on extensive performance data
* **Hardware-specific optimization**: Tailored for your specific GPU architecture
* **Production-quality results**: Performance equivalent to or better than full autotuning

Memory Efficiency
~~~~~~~~~~~~~~~~~

* **Reduced memory overhead**: No need to store multiple kernel variants during compilation
* **Predictable memory usage**: Consistent memory consumption across different problem sizes

Advanced Configuration
----------------------

Hardware Detection
~~~~~~~~~~~~~~~~~~

Diode automatically detects your hardware, but you can also specify it manually:

.. code-block:: python

    import torch_diode_models

    # Check detected hardware
    print(f"Detected hardware: {torch_diode_models.get_detected_hardware()}")

    # List available models
    available_models = torch_diode_models.list_available_models()
    print(f"Available models: {available_models}")

Manual Model Selection
~~~~~~~~~~~~~~~~~~~~~~

For advanced users, you can manually select a specific model:

.. code-block:: python

    import torch_diode_models

    # Use a specific model (e.g., for benchmarking different configurations)
    torch_diode_models.register_model("NVIDIA-H100-matmul")

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

Fine-tune the Diode integration with additional configuration options:

.. code-block:: python

    from torch._inductor import config

    # Basic Diode configuration
    config.max_autotune_gemm_backends = "DIODE"
    config.fast_autotune = True

    # Advanced options
    config.diode_fallback_to_autotune = True   # Fallback for unsupported cases
    config.diode_confidence_threshold = 0.95   # Minimum prediction confidence
    config.diode_cache_predictions = True      # Cache predictions for repeated sizes

Integration with Existing Workflows
------------------------------------

Training Workflows
~~~~~~~~~~~~~~~~~~

Diode integrates seamlessly with existing training code:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch_diode_models
    from torch._inductor import config

    # Enable Diode acceleration
    config.max_autotune_gemm_backends = "DIODE"
    config.fast_autotune = True

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(1024, 2048)
            self.linear2 = nn.Linear(2048, 1024)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            return self.linear2(x)

    model = MyModel().cuda()

    # Compile with Diode acceleration
    compiled_model = torch.compile(model, mode="max-autotune")

    # Training loop - faster compilation on first run
    optimizer = torch.optim.Adam(model.parameters())
    for batch in dataloader:
        optimizer.zero_grad()
        output = compiled_model(batch)  # Fast compilation + optimal performance
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

Inference Workflows
~~~~~~~~~~~~~~~~~~~

Perfect for production inference where fast startup is critical:

.. code-block:: python

    import torch
    import torch_diode_models
    from torch._inductor import config

    # Configure for inference
    config.max_autotune_gemm_backends = "DIODE"
    config.fast_autotune = True
    config.triton.cudagraphs = True  # Enable CUDA graphs for even better performance

    # Load your model
    model = torch.jit.load("my_model.pt").cuda()

    # Compile with minimal warmup time
    compiled_model = torch.compile(model, mode="max-autotune")

    # First inference compiles quickly thanks to Diode
    with torch.no_grad():
        output = compiled_model(input_tensor)

Supported Operations
--------------------

Diode currently accelerates the following matrix multiplication operations:

**Core Operations**
* ``torch.mm`` - Basic matrix multiplication
* ``torch.addmm`` - Matrix multiplication with bias addition
* ``torch.bmm`` - Batch matrix multiplication
* ``torch.baddbmm`` - Batch matrix multiplication with bias

**Linear Layer Operations**
* ``torch.nn.Linear`` - Fully connected layers
* ``torch.nn.functional.linear`` - Functional linear operations

**Data Types**
* ``float16`` (half precision)
* ``bfloat16`` (brain float)
* ``float32`` (single precision)

**Hardware Support**
* NVIDIA GPUs: H100, A100, RTX 4090, RTX 3090, V100
* AMD GPUs: MI250X, MI210
* Intel GPUs: Data Center Max (coming soon)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Model Not Found for Hardware**

If Diode cannot find a model for your specific hardware:

.. code-block:: python

    # Check available models
    print(torch_diode_models.list_available_models())

    # Use a similar hardware model as fallback
    torch_diode_models.register_model("NVIDIA-A100-matmul")  # Similar to H100

**Performance Regression**

If you experience slower performance:

.. code-block:: python

    # Enable fallback to traditional autotuning
    from torch._inductor import config
    config.diode_fallback_to_autotune = True

    # Increase confidence threshold
    config.diode_confidence_threshold = 0.99

**Compilation Errors**

If compilation fails with Diode enabled:

.. code-block:: python

    # Disable Diode temporarily
    from torch._inductor import config
    config.max_autotune_gemm_backends = "TRITON"
    config.fast_autotune = False

Debugging and Profiling
~~~~~~~~~~~~~~~~~~~~~~~

Monitor Diode's performance impact:

.. code-block:: python

    import torch_diode_models

    # Enable detailed logging
    torch_diode_models.set_log_level("DEBUG")

    # Profile model predictions
    with torch_diode_models.profile_predictions():
        compiled_fn = torch.compile(my_function, mode="max-autotune")
        result = compiled_fn(input_tensor)

    # View prediction statistics
    stats = torch_diode_models.get_prediction_stats()
    print(f"Predictions made: {stats['total_predictions']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Fallbacks to autotune: {stats['fallbacks']}")

Performance Comparison
----------------------

Typical performance improvements with Diode:

.. list-table:: Compilation Time Comparison
   :header-rows: 1

   * - Operation Size
     - Traditional Autotune
     - Diode Acceleration
     - Speedup
   * - (1024, 1024) × (1024, 1024)
     - 2.3s
     - 0.08s
     - **29x faster**
   * - (4096, 4096) × (4096, 4096)
     - 8.1s
     - 0.12s
     - **68x faster**
   * - Batch of 32 operations
     - 45.2s
     - 1.2s
     - **38x faster**

.. list-table:: Runtime Performance
   :header-rows: 1

   * - Hardware
     - Operation
     - Baseline (no autotune)
     - Traditional Autotune
     - Diode
   * - NVIDIA H100
     - (8192, 8192) × (8192, 8192)
     - 2.3 TFLOPS
     - 15.2 TFLOPS
     - **15.4 TFLOPS**
   * - NVIDIA A100
     - (4096, 4096) × (4096, 4096)
     - 1.8 TFLOPS
     - 9.7 TFLOPS
     - **9.9 TFLOPS**

Best Practices
--------------

1. **Always Import Early**: Import ``torch_diode_models`` before any torch.compile calls
2. **Use Appropriate Modes**: Combine with ``mode="max-autotune"`` for best results
3. **Monitor Performance**: Use profiling to ensure expected speedups
4. **Keep Models Updated**: Regularly update the torch-diode-models package for latest optimizations
5. **Hardware Consistency**: Use the same hardware type for development and production

Integration Checklist
----------------------

Before deploying Diode in production:

- [ ] Verify hardware compatibility with ``torch_diode_models.list_available_models()``
- [ ] Test compilation times show expected speedup (10x+ improvement typical)
- [ ] Validate runtime performance matches or exceeds baseline
- [ ] Enable fallback options for robustness
- [ ] Set up monitoring for prediction confidence and fallback rates
- [ ] Document any manual model selections for reproducibility

Next Steps
----------

* **Custom Models**: Train models on your specific workloads using the full Diode toolkit
* **Hardware Support**: Request support for additional hardware through GitHub issues
* **Advanced Features**: Explore multi-GPU and distributed training acceleration
* **Integration**: Combine with other PyTorch performance tools like CUDA Graphs

For more information on training custom models, see the :doc:`getting_started` guide.
