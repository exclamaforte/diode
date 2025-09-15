#!/usr/bin/env python3
"""
Test runner for the Diode PyTorch Inductor integration tests.

This script runs all the integration tests and provides a summary of the results.
"""

import os
import sys
import unittest
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_unit_tests():
    """Run the unit tests for the integration."""
    print("=" * 60)
    print("Running Unit Tests for Diode Inductor Integration")
    print("=" * 60)
    
    # Import the test module
    from tests.integration.test_inductor_integration import (
        TestDiodeInductorChoices,
        TestFactoryFunctions,
        TestErrorHandling
    )
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in [TestDiodeInductorChoices, TestFactoryFunctions, TestErrorHandling]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run the end-to-end integration tests."""
    print("\n" + "=" * 60)
    print("Running End-to-End Integration Tests")
    print("=" * 60)
    
    # Import the test module
    from tests.integration.test_inductor_integration_end_to_end import (
        TestEndToEndIntegration,
        TestIntegrationWithMockedInductor,
        TestRealWorldSimulation
    )
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in [TestEndToEndIntegration, TestIntegrationWithMockedInductor, TestRealWorldSimulation]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_pattern):
    """Run specific tests matching the pattern."""
    print(f"\n" + "=" * 60)
    print(f"Running Tests Matching Pattern: {test_pattern}")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern=f"*{test_pattern}*.py",
        top_level_dir=str(project_root)
    )
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests():
    """Run all integration tests."""
    print("Starting Diode PyTorch Inductor Integration Test Suite")
    print("=" * 80)
    
    success = True
    
    # Run unit tests
    try:
        unit_success = run_unit_tests()
        success = success and unit_success
    except Exception as e:
        print(f"Error running unit tests: {e}")
        success = False
    
    # Run integration tests
    try:
        integration_success = run_integration_tests()
        success = success and integration_success
    except Exception as e:
        print(f"Error running integration tests: {e}")
        success = False
    
    # Print summary
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 80)
    
    return success


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run Diode PyTorch Inductor Integration Tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true", 
        help="Run only end-to-end integration tests"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        help="Run tests matching the specified pattern"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    success = True
    
    if args.pattern:
        success = run_specific_test(args.pattern)
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    else:
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()