# Torch-Diode Systems Architecture

This diagram shows how the torch-diode system components interact with PyTorch Inductor to provide ML-based kernel selection.

```mermaid
graph TB
    %% User Code and Entry Points
    subgraph "User Code"
        UC[User Code using torch.compile]
        IMPORT[import diode]
    end

    %% PyTorch Inductor
    subgraph "PyTorch Inductor"
        VIRT[V - Virtualized Environment]
        CHOICES[InductorChoices]
        MM[MM Kernel Selection]
        AUTOTUNE[Autotune Process]
        KERNEL[Kernel Templates]
    end

    %% Diode Core System
    subgraph "Diode Core"
        INIT[diode.__init__.py<br/>Auto-registration System]
        REGISTRY[Integration Registry]
        MODEL_REG[Model Registry]
        STATUS[Status Tracking]
    end

    %% Integration Layer
    subgraph "Integration Layer"
        BASE_INT[BaseIntegration<br/>Abstract Base]
        MATMUL_INT[MatmulIntegration<br/>Concrete Implementation]
        INDUCTOR_INT[DiodeInductorChoices<br/>Extends InductorChoices]
        DUMMY[Dummy Functions<br/>Interface Testing]
    end

    %% Model Management
    subgraph "Model Management"
        MODEL_PTR[ModelPointer<br/>Model Metadata]
        MODEL_WRAP[ModelWrapper<br/>Model Loading/Inference]
        TIMING_MODEL[MatmulTimingModel<br/>Neural Network]
        CONFIG[Model Configuration]
    end

    %% Data Collection System
    subgraph "Data Collection"
        COLLECTOR[MatmulDatasetCollector<br/>Hooks into Feedback]
        FEEDBACK[Feedback Saver Interface]
        DATASET[Dataset<br/>Timing Data Storage]
        FEATURES[Feature Extraction<br/>Problem + Config Features]
    end

    %% Type System
    subgraph "Type System"
        MM_SHAPE[MMShape<br/>Problem Definition]
        TRITON_CONFIG[TritonGEMMConfig<br/>Kernel Configuration]
        DATASET_TYPES[Dataset Types]
    end

    %% File System
    subgraph "Storage"
        TRAINED_MODELS[(trained_models/<br/>*.pt files)]
        MODEL_CONFIG[(Model Configs<br/>*.json files)]
        COLLECTED_DATA[(Collected Data<br/>Timing Results)]
    end

    %% Workflows and Utilities
    subgraph "Workflows & Utils"
        WORKFLOWS[Workflows<br/>Training/Validation Scripts]
        UTILS[Utilities<br/>Dataset Utils<br/>Visualization]
        BUILD[Build System<br/>Package Creation]
    end

    %% Connections - Auto-registration Flow
    IMPORT --> INIT
    INIT --> |1. Discover| REGISTRY
    INIT --> |2. Load Models| MODEL_REG
    REGISTRY --> |3. Create| MATMUL_INT
    MATMUL_INT --> |4. Test Interface| DUMMY
    DUMMY --> |5. Register Dummy| VIRT
    MATMUL_INT --> |6. Load Model| MODEL_PTR
    MODEL_PTR --> |7. Load from disk| TRAINED_MODELS
    TRAINED_MODELS --> MODEL_WRAP
    MODEL_WRAP --> |8. Create DiodeChoices| INDUCTOR_INT
    INDUCTOR_INT --> |9. Register| VIRT
    VIRT --> |10. Enable Configs| CHOICES

    %% Runtime Flow - Model-based Selection
    UC --> |torch.compile| AUTOTUNE
    AUTOTUNE --> MM
    MM --> CHOICES
    CHOICES --> |Override _finalize_mm_configs| INDUCTOR_INT
    INDUCTOR_INT --> |Extract features| FEATURES
    FEATURES --> MM_SHAPE
    FEATURES --> TRITON_CONFIG
    INDUCTOR_INT --> |Predict performance| MODEL_WRAP
    MODEL_WRAP --> TIMING_MODEL
    TIMING_MODEL --> |Return predictions| INDUCTOR_INT
    INDUCTOR_INT --> |Select top configs| KERNEL
    KERNEL --> |Execute| UC

    %% Data Collection Flow
    AUTOTUNE --> |Timing feedback| FEEDBACK
    COLLECTOR --> |Hook into| FEEDBACK
    FEEDBACK --> |Store results| DATASET
    DATASET --> |Training data| WORKFLOWS
    WORKFLOWS --> |Train new models| TRAINED_MODELS

    %% Configuration and Metadata
    MODEL_PTR --> MODEL_CONFIG
    MODEL_CONFIG --> CONFIG
    MODEL_REG --> |Manifest| BUILD
    STATUS --> |Monitor| INIT

    %% Support Systems
    UTILS --> FEATURES
    UTILS --> DATASET
    WORKFLOWS --> MODEL_WRAP
    BUILD -->|Package| TRAINED_MODELS

    %% Styling
    classDef userCode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef pytorch fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef diodeCore fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef integration fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef model fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    classDef data fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef types fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef storage fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef workflows fill:#fafafa,stroke:#424242,stroke-width:2px

    class UC,IMPORT userCode
    class VIRT,CHOICES,MM,AUTOTUNE,KERNEL pytorch
    class INIT,REGISTRY,MODEL_REG,STATUS diodeCore
    class BASE_INT,MATMUL_INT,INDUCTOR_INT,DUMMY integration
    class MODEL_PTR,MODEL_WRAP,TIMING_MODEL,CONFIG model
    class COLLECTOR,FEEDBACK,DATASET,FEATURES data
    class MM_SHAPE,TRITON_CONFIG,DATASET_TYPES types
    class TRAINED_MODELS,MODEL_CONFIG,COLLECTED_DATA storage
    class WORKFLOWS,UTILS,BUILD workflows
```

## Key Architecture Components

### 1. **Auto-registration System** (Main Entry Point)
- `diode.__init__.py` orchestrates the entire integration process
- Discovers available integrations automatically
- Tests PyTorch interfaces with dummy functions
- Loads actual models only when interfaces are available
- Enables relevant PyTorch configurations

### 2. **Integration Layer** (PyTorch Interface)
- `BaseIntegration`: Abstract framework for all integrations
- `MatmulIntegration`: Specific implementation for matrix multiplication
- `DiodeInductorChoices`: Extends PyTorch's InductorChoices to inject ML predictions
- Seamlessly overrides `_finalize_mm_configs` method

### 3. **Model Management System**
- `ModelPointer`: Metadata and path management for trained models
- `ModelWrapper`: Loads, compiles, and runs inference on trained models
- `ModelRegistry`: Centralized registry of all available models
- Supports multiple model formats and configurations

### 4. **Data Collection Pipeline**
- `MatmulDatasetCollector`: Hooks into PyTorch's feedback system
- Collects timing data during autotune processes
- Extracts features from problem shapes and kernel configurations
- Stores structured datasets for model training

### 5. **Type System**
- `MMShape`: Represents matrix multiplication problem characteristics
- `TritonGEMMConfig`: Represents kernel configuration parameters
- Strong typing ensures data consistency across the system

### 6. **Inductor Integration Flow**
1. User imports diode → auto-registration begins
2. System tests PyTorch interfaces with dummy functions
3. Loads trained models from disk if interfaces are available
4. Registers `DiodeInductorChoices` as the active choice handler
5. During `torch.compile`, the system intercepts kernel selection
6. Extracts features from the problem and available configurations
7. Runs ML model inference to predict kernel performance
8. Returns top-k configurations based on predictions

### 7. **Data Flow**
- **Training Time**: Collector gathers timing data → Workflows train models → Models stored to disk
- **Inference Time**: Models loaded → Features extracted → Predictions made → Best kernels selected

This architecture allows torch-diode to seamlessly integrate with PyTorch's compilation pipeline while maintaining modularity and extensibility for future optimization targets beyond matrix multiplication.
