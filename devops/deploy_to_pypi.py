#!/usr/bin/env python3
"""
Script to build and deploy torch-diode packages to PyPI.

This script builds both torch-diode packages (torch-diode and torch-diode-lib)
and uploads them to either TestPyPI or production PyPI based on the environment argument.

torch-diode: includes all code + auto-registers to PyTorch Inductor
torch-diode-lib: same code but without auto-registration functionality

Usage:
    python deploy_to_pypi.py test    # Deploy to TestPyPI
    python deploy_to_pypi.py prod    # Deploy to production PyPI

Requirements:
    - twine installed (pip install twine)
    - build installed (pip install build)
    - Appropriate PyPI credentials configured

For TestPyPI:
    - Set TESTPYPI_USERNAME and TESTPYPI_PASSWORD environment variables, or
    - Configure ~/.pypirc with [testpypi] section

For Production PyPI:
    - Set PYPI_USERNAME and PYPI_PASSWORD environment variables, or
    - Configure ~/.pypirc with [pypi] section
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


class Colors:
    """ANSI color codes for terminal output."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"  # End color
    BOLD = "\033[1m"


def print_step(message: str) -> None:
    """Print a step message in blue."""
    print(f"{Colors.BLUE}{Colors.BOLD}==> {message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def run_command(
    cmd: List[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    if cwd:
        print(f"  in directory: {cwd}")

    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd_str}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def check_dependencies() -> None:
    """Check if required tools are installed."""
    print_step("Checking dependencies")

    # Check for build
    try:
        run_command(["python", "-m", "build", "--help"], check=True)
        print_success("build is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("build is not installed. Install with: pip install build")
        sys.exit(1)

    # Check for twine
    try:
        run_command(["twine", "--version"], check=True)
        print_success("twine is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("twine is not installed. Install with: pip install twine")
        sys.exit(1)


def clean_build_artifacts(package_dir: Path) -> None:
    """Clean existing build artifacts."""
    print_step(f"Cleaning build artifacts in {package_dir.name}")

    artifacts = ["build", "dist", "*.egg-info"]
    for pattern in artifacts:
        if pattern.startswith("*"):
            # Handle glob patterns
            import glob

            for path in glob.glob(str(package_dir / pattern)):
                path_obj = Path(path)
                if path_obj.exists():
                    if path_obj.is_dir():
                        shutil.rmtree(path_obj)
                        print(f"Removed directory: {path_obj}")
                    else:
                        path_obj.unlink()
                        print(f"Removed file: {path_obj}")
        else:
            artifact_path = package_dir / pattern
            if artifact_path.exists():
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                    print(f"Removed directory: {artifact_path}")
                else:
                    artifact_path.unlink()
                    print(f"Removed file: {artifact_path}")


def build_torch_diode_packages(base_dir: Path) -> None:
    """Build both torch-diode packages using Make."""
    print_step("Building both torch-diode and torch-diode-lib packages")

    # Clean first
    clean_build_artifacts(base_dir)

    # Use make to build both packages
    run_command(["make", "build-all"], cwd=base_dir)
    print_success("Successfully built both torch-diode packages")


def upload_packages_from_dist(base_dir: Path, repository: str) -> None:
    """Upload all packages from the dist directory to PyPI."""
    print_step(f"Uploading packages to {repository}")

    dist_dir = base_dir / "dist"
    if not dist_dir.exists() or not list(dist_dir.glob("*")):
        print_error(f"No distribution files found in {dist_dir}")
        return

    # List what we're uploading
    dist_files = list(dist_dir.glob("*"))
    print("Found distribution files:")
    for file in dist_files:
        print(f"  - {file.name}")

    # Upload using twine
    cmd = ["twine", "upload", "--repository", repository, "dist/*"]
    run_command(cmd, cwd=base_dir)
    print_success(f"Successfully uploaded packages to {repository}")


def verify_torch_diode_setup(base_dir: Path) -> bool:
    """Verify that the torch-diode project structure is correct."""
    required_files = [
        "pyproject.toml",  # Main torch-diode config
        "pyproject-lib.toml",  # torch-diode-lib config
        "Makefile",  # Build system
        "diode/__init__.py",  # Main package init
        "diode/__init___lib.py",  # Library package init
    ]

    for file_path in required_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            print_error(f"Required file not found: {file_path}")
            return False

    return True


def check_pypi_credentials(environment: str) -> None:
    """Check if PyPI credentials are available."""
    print_step(f"Checking {environment} PyPI credentials")

    if environment == "test":
        # Check for TestPyPI credentials
        if not (os.getenv("TESTPYPI_USERNAME") and os.getenv("TESTPYPI_PASSWORD")):
            pypirc_path = Path.home() / ".pypirc"
            if not pypirc_path.exists():
                print_warning(
                    "No TestPyPI credentials found in environment variables or ~/.pypirc"
                )
                print(
                    "Set TESTPYPI_USERNAME and TESTPYPI_PASSWORD environment variables"
                )
                print("Or configure ~/.pypirc with [testpypi] section")
            else:
                print_success("Found ~/.pypirc file")
        else:
            print_success("Found TestPyPI credentials in environment variables")
    else:
        # Check for production PyPI credentials
        if not (os.getenv("PYPI_USERNAME") and os.getenv("PYPI_PASSWORD")):
            pypirc_path = Path.home() / ".pypirc"
            if not pypirc_path.exists():
                print_warning(
                    "No PyPI credentials found in environment variables or ~/.pypirc"
                )
                print("Set PYPI_USERNAME and PYPI_PASSWORD environment variables")
                print("Or configure ~/.pypirc with [pypi] section")
            else:
                print_success("Found ~/.pypirc file")
        else:
            print_success("Found PyPI credentials in environment variables")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build and deploy torch-diode packages to PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "environment",
        choices=["test", "prod"],
        help="Target environment: test (TestPyPI) or prod (production PyPI)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building packages (use existing dist files)",
    )
    parser.add_argument(
        "--build-only", action="store_true", help="Only build packages, do not upload"
    )

    args = parser.parse_args()

    # Determine repository
    repository = "testpypi" if args.environment == "test" else "pypi"

    print(f"{Colors.BOLD}Torch-Diode PyPI Deployment Script{Colors.ENDC}")
    print(f"Environment: {args.environment}")
    print(f"Repository: {repository}")
    print()

    # Get the base directory (should be /diode)
    base_dir = Path(__file__).parent.absolute()
    if base_dir.name != "diode":
        print_error(f"Script should be run from the diode directory, got: {base_dir}")
        sys.exit(1)

    # Verify torch-diode project structure
    if not verify_torch_diode_setup(base_dir):
        print_error("torch-diode project structure is not correct")
        sys.exit(1)
    print_success("torch-diode project structure verified")

    # Check dependencies
    if not args.skip_build:
        check_dependencies()

    # Check credentials if not build-only
    if not args.build_only:
        check_pypi_credentials(args.environment)

    print_step("Package details:")
    print("  - torch-diode: includes all code + auto-registers to PyTorch Inductor")
    print("  - torch-diode-lib: same code but without auto-registration functionality")
    print()

    # Build packages
    if not args.skip_build:
        try:
            build_torch_diode_packages(base_dir)
        except subprocess.CalledProcessError:
            print_error("Failed to build torch-diode packages")
            sys.exit(1)
        print()

    # Upload packages
    if not args.build_only:
        try:
            upload_packages_from_dist(base_dir, repository)
        except subprocess.CalledProcessError:
            print_error("Failed to upload torch-diode packages")
            sys.exit(1)

    print()
    if args.build_only:
        print_success("Both torch-diode packages built successfully!")
        print("Distribution files are available in the dist/ directory:")
        dist_dir = base_dir / "dist"
        if dist_dir.exists():
            dist_files = list(dist_dir.glob("*"))
            for file in dist_files:
                print(f"  - {file.name}")
    else:
        print_success(f"Deployment to {repository} completed!")
        if args.environment == "test":
            print("You can test installation with:")
            print("  pip install --index-url https://test.pypi.org/simple/ torch-diode")
            print(
                "  pip install --index-url https://test.pypi.org/simple/ torch-diode-lib"
            )
        else:
            print("You can install the packages with:")
            print("  pip install torch-diode")
            print("  pip install torch-diode-lib")


if __name__ == "__main__":
    main()
