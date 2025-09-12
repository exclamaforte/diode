"""
Example usage of the Diode PyTorch Inductor integration.

This module demonstrates how to use the Diode integration with PyTorch Inductor
to enable model-based kernel configuration selection.
"""

import torch
import logging
from typing import Optional

# Import the integration module
from diode.integration.inductor_integration import (
    DiodeInductorChoices,
    install_diode_choices,
    create_diode_choices
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_diode_integration(
    model_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    top_k_configs: int = 3,
    performance_threshold: float = 1.1
):
    """
    Set up Diode integration with PyTorch Inductor.
    
    Args:
        model_path: Path to the trained Diode model (optional)
        device: Device to run the model on
        top_k_configs: Maximum number of configurations to return
        performance_threshold: Performance threshold for config selection
    """
    logger.info("Setting up Diode integration with PyTorch Inductor...")
    
    try:
        # Install Diode choices as the default choice handler
        install_diode_choices(
            model_path=model_path,
            device=device,
            top_k_configs=top_k_configs,
            performance_threshold=performance_threshold
        )
        
        logger.info("Diode integration setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to setup Diode integration: {e}")
        raise


def run_matrix_multiplication_example():
    """
    Run a simple matrix multiplication example to demonstrate the integration.
    """
    logger.info("Running matrix multiplication example...")
    
    # Create sample matrices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Example 1: Regular matrix multiplication
    A = torch.randn(512, 256, device=device, dtype=torch.float16)
    B = torch.randn(256, 512, device=device, dtype=torch.float16)
    
    # Compile the operation (this will use Diode model for config selection)
    @torch.compile(mode="max-autotune")
    def matmul_op(a, b):
        return torch.mm(a, b)
    
    # Run the compiled function
    result = matmul_op(A, B)
    logger.info(f"Matrix multiplication result shape: {result.shape}")
    
    # Example 2: Batch matrix multiplication
    A_batch = torch.randn(4, 128, 64, device=device, dtype=torch.float16)
    B_batch = torch.randn(4, 64, 128, device=device, dtype=torch.float16)
    
    @torch.compile(mode="max-autotune")
    def bmm_op(a, b):
        return torch.bmm(a, b)
    
    result_batch = bmm_op(A_batch, B_batch)
    logger.info(f"Batch matrix multiplication result shape: {result_batch.shape}")
    
    # Example 3: Matrix multiplication with addition (addmm)
    C = torch.randn(256, 512, device=device, dtype=torch.float16)
    
    @torch.compile(mode="max-autotune")
    def addmm_op(c, a, b):
        return torch.addmm(c, a, b)
    
    result_addmm = addmm_op(C, A, B)
    logger.info(f"Matrix multiplication with addition result shape: {result_addmm.shape}")
    
    logger.info("Matrix multiplication examples completed successfully!")


def demonstrate_manual_usage():
    """
    Demonstrate manual usage of DiodeInductorChoices without installing globally.
    """
    logger.info("Demonstrating manual usage of DiodeInductorChoices...")
    
    # Create a DiodeInductorChoices instance manually
    choices = create_diode_choices(
        model_path="matmul_model_exhaustive.pt",  # Will try to find automatically if None
        device="cuda" if torch.cuda.is_available() else "cpu",
        top_k_configs=5,
        performance_threshold=1.2  # Allow configs within 20% of best
    )
    
    # You can use this instance directly or install it temporarily
    # For now, we'll just demonstrate that it was created successfully
    stats = choices.get_stats()
    logger.info(f"DiodeInductorChoices created successfully. Initial stats: {stats}")
    
    # You could temporarily install it like this:
    # from torch._inductor.virtualized import V
    # old_handler = V.get_choices_handler()
    # V.set_choices_handler(choices)
    # ... run your operations ...
    # V.set_choices_handler(old_handler)  # Restore original handler


def check_integration_status():
    """
    Check if Diode integration is properly installed and working.
    """
    logger.info("Checking Diode integration status...")
    
    try:
        from torch._inductor.virtualized import V
        
        # Get the current choices handler
        current_handler = V.get_choices_handler()
        
        if isinstance(current_handler, DiodeInductorChoices):
            logger.info("✓ Diode integration is active")
            stats = current_handler.get_stats()
            logger.info(f"  Model path: {current_handler.model_path}")
            logger.info(f"  Model loaded: {current_handler._model_loaded}")
            logger.info(f"  Device: {current_handler.device}")
            logger.info(f"  Top-k configs: {current_handler.top_k_configs}")
            logger.info(f"  Performance threshold: {current_handler.performance_threshold}")
            logger.info(f"  Stats: {stats}")
        else:
            logger.info("✗ Diode integration is not active")
            logger.info(f"  Current handler type: {type(current_handler)}")
    
    except ImportError:
        logger.error("Could not import PyTorch Inductor virtualized module")
    except Exception as e:
        logger.error(f"Error checking integration status: {e}")


def main():
    """
    Main function to demonstrate Diode integration usage.
    """
    logger.info("=== Diode PyTorch Inductor Integration Example ===")
    
    # Step 1: Check initial status
    check_integration_status()
    
    # Step 2: Setup Diode integration
    try:
        setup_diode_integration(
            # model_path="path/to/your/model.pt",  # Specify your model path here
            device="cuda" if torch.cuda.is_available() else "cpu",
            top_k_configs=3,
            performance_threshold=1.1
        )
    except Exception as e:
        logger.warning(f"Could not setup integration (may not be in Inductor environment): {e}")
        logger.info("Continuing with manual usage demonstration...")
    
    # Step 3: Check status after setup
    check_integration_status()
    
    # Step 4: Demonstrate manual usage
    demonstrate_manual_usage()
    
    # Step 5: Run matrix multiplication examples (if in proper environment)
    try:
        run_matrix_multiplication_example()
    except Exception as e:
        logger.warning(f"Could not run matrix multiplication examples: {e}")
        logger.info("This is expected if not running in a proper PyTorch Inductor environment")
    
    logger.info("=== Example completed ===")


if __name__ == "__main__":
    main()