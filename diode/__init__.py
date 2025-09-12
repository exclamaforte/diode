"""Framework for training ML diodeistics in Torch."""

__version__ = "0.1.0"

# Auto-register with PyTorch Inductor when the package is imported
try:
    import logging

    from diode.integration.inductor_integration import install_diode_choices

    logger = logging.getLogger(__name__)

    # Attempt to install Diode choices with fallback behavior enabled
    try:
        install_diode_choices(enable_fallback=True)
        logger.info("Diode successfully registered with PyTorch Inductor")
    except Exception as e:
        logger.debug(
            f"Diode registration with PyTorch Inductor failed (expected in some environments): {e}"
        )
        # This is expected if PyTorch Inductor is not available or if we're not in the right environment
        pass

except ImportError:
    # Expected when running without PyTorch or in environments where Inductor is not available
    pass
