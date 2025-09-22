Accelerating torch.compile with Diode
=====================================

Diode can speeds up PyTorch's ``torch.compile`` by providing pre-trained models that predict optimal matrix multiplication configurations, eliminating the need for expensive runtime autotuning. This integration allows you to get the performance benefits of extensive autotuning with minimal compilation time.

Overview
--------

When PyTorch compiles matrix multiplication operations, it typically needs to search through many different kernel configurations to find the optimal one for your specific hardware and problem size. This process, called autotuning, can take substantial time during compilation.

Diode solves this by providing a pre-trained model that predicts the optimal configuration for a given hardware and problem size. This saves compilation time by eliminating the need for runtime autotuning, while still providing optimal performance.

Quick Start
-----------

Getting started with Diode acceleration is simple and requires only three steps:

Step 1: Install torch-diode-models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the pre-trained models package:

.. code-block:: bash

    pip install torch-diode

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
    config.max_autotune = True

Complete Example
----------------

Here's a complete example showing how to use Diode with torch.compile:

.. code-block:: python

    import torch
    import torch_diode_models  # Auto-registers the best model for your hardware
    from torch._inductor import config

    # Configure PyTorch to use Diode acceleration
    config.max_autotune_gemm_backends = "DIODE"
    config.max_autotune = True

    # Your existing PyTorch code - no changes needed!
    def matmul_function(a, b):
        return torch.mm(a, b)

    # Compile with torch.compile - now accelerated by Diode
    compiled_fn = torch.compile(matmul_function, mode="max-autotune")

    # Use as normal
    a = torch.randn(1024, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)

    result = compiled_fn(a, b)

Benefits
--------

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

Diode provides significant improvements in both compilation time and runtime performance:

**Compilation Speed**
* **10x faster compilation**: Eliminates expensive autotuning searches
* **Instant predictions**: Model inference takes microseconds vs. seconds of autotuning
* **Consistent compile times**: No variation based on problem size or hardware load

**Runtime Performance**
* **Max Autotune**: We can match Max Autotune and Max Autotune EXHAUSTIVE performance.
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
    config.max_autotune = True

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

**Data Types**
* ``float16`` (half precision)
* ``bfloat16`` (brain float)
* ``float32`` (single precision)

**Hardware Support**
* NVIDIA GPUs: H100
* AMD GPUs: MI210

For more information on training custom models, see the :doc:`getting_started` guide.
