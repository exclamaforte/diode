"""Example module to demonstrate docstring formats for Sphinx documentation.

This module contains examples of Google-style docstrings that can be parsed by
Sphinx using the napoleon extension. Use these examples as templates for
documenting your own code.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ExampleClass:
    """Example class with Google-style docstrings.

    This class demonstrates how to write Google-style docstrings that can be
    parsed by Sphinx using the napoleon extension.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (int): Description of `attr2`.
    """

    def __init__(self, attr1: str, attr2: int = 42):
        """Initialize the ExampleClass.

        Args:
            attr1: Description of `attr1`.
            attr2: Description of `attr2`. Defaults to 42.
        """
        self.attr1 = attr1
        self.attr2 = attr2

    def method_with_types(self, param1: str, param2: int) -> bool:
        """Example method with type annotations.

        This method demonstrates how to document a method with type annotations.

        Args:
            param1: Description of `param1`.
            param2: Description of `param2`.

        Returns:
            True if successful, False otherwise.

        Raises:
            ValueError: If `param2` is negative.
        """
        if param2 < 0:
            raise ValueError("param2 cannot be negative")
        return True

    def method_with_complex_types(
        self,
        param1: List[Dict[str, Union[int, str]]],
        param2: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Example method with complex type annotations.

        This method demonstrates how to document a method with complex type
        annotations.

        Args:
            param1: List of dictionaries mapping strings to either integers or
                strings. Each dictionary represents a data point.
            param2: Optional numpy array. If provided, it will be processed.
                Defaults to None.

        Returns:
            A tuple containing:
                - A numpy array of processed data
                - A dictionary mapping feature names to their importance scores

        Raises:
            ValueError: If `param1` is empty.
            TypeError: If `param2` is provided but not a numpy array.
        """
        if not param1:
            raise ValueError("param1 cannot be empty")
        if param2 is not None and not isinstance(param2, np.ndarray):
            raise TypeError("param2 must be a numpy array")
        
        # Placeholder implementation
        return np.array([1, 2, 3]), {"feature1": 0.8, "feature2": 0.2}


def function_with_examples(x: float, y: float) -> float:
    """Calculate the sum of squares of two numbers.

    This function demonstrates how to include examples in docstrings.

    Args:
        x: First number.
        y: Second number.

    Returns:
        The sum of squares of x and y.

    Examples:
        >>> function_with_examples(3, 4)
        25.0
        
        >>> function_with_examples(0, 0)
        0.0
    """
    return x**2 + y**2


def function_with_notes_and_warnings(data: List[float]) -> float:
    """Calculate the average of a list of numbers.

    This function demonstrates how to include notes and warnings in docstrings.

    Args:
        data: List of numbers.

    Returns:
        The average of the numbers in the list.

    Notes:
        This function does not handle empty lists gracefully.
        
        The implementation uses the built-in `sum` function for efficiency.

    Warnings:
        This function will raise a ZeroDivisionError if the list is empty.
    """
    return sum(data) / len(data)


def function_with_math(x: float) -> float:
    """Calculate the value of a mathematical expression.

    This function demonstrates how to include mathematical expressions in
    docstrings using LaTeX syntax.

    The function calculates:
    
    .. math::
        f(x) = \\frac{x^2 + 2x + 1}{x + 1}
    
    which simplifies to:
    
    .. math::
        f(x) = x + 1
    
    Args:
        x: Input value.

    Returns:
        The result of the mathematical expression.
        
    Raises:
        ZeroDivisionError: If x = -1.
    """
    return (x**2 + 2*x + 1) / (x + 1)
