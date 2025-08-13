#!/usr/bin/env python3
"""
Python formatting linter adapter for lintrunner (usort + ruff-format).
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


def run_pyfmt(files: List[str], apply_fixes: bool = False) -> List[Dict[str, Any]]:
    """Run Python formatting tools on the given files and return results."""
    if not files:
        return []
    
    # Filter out files that don't exist or are temporary files
    valid_files = []
    for file_path in files:
        if file_path.startswith('@/tmp/'):
            continue  # Skip temporary files that cause issues
        try:
            import os
            if os.path.exists(file_path):
                valid_files.append(file_path)
        except:
            continue
    
    if not valid_files:
        return []
    
    violations = []
    
    # Run usort (import sorting)
    try:
        if apply_fixes:
            # Apply usort fixes
            cmd = [sys.executable, "-m", "usort"] + valid_files
            subprocess.run(cmd, check=False)
        else:
            # Check only
            cmd = [sys.executable, "-m", "usort", "--check", "--diff"] + valid_files
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                for file_path in valid_files:
                    violations.append({
                        "path": file_path,
                        "line": 1,
                        "char": 1,
                        "code": "USORT",
                        "severity": "error",
                        "name": "PYFMT",
                        "original": None,
                        "replacement": None,
                        "description": "Import sorting issues detected by usort",
                    })
    except subprocess.CalledProcessError:
        pass
    
    # Run ruff format
    try:
        if apply_fixes:
            # Apply ruff format fixes
            cmd = [sys.executable, "-m", "ruff", "format"] + valid_files
            subprocess.run(cmd, check=False)
        else:
            # Check only
            cmd = [sys.executable, "-m", "ruff", "format", "--check", "--diff"] + valid_files
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                for file_path in valid_files:
                    violations.append({
                        "path": file_path,
                        "line": 1,
                        "char": 1,
                        "code": "RUFF_FORMAT",
                        "severity": "error",
                        "name": "PYFMT",
                        "original": None,
                        "replacement": None,
                        "description": "Code formatting issues detected by ruff format",
                    })
    except subprocess.CalledProcessError:
        pass
    
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python formatting linter")
    parser.add_argument("files", nargs="*", help="Files to lint")
    
    args = parser.parse_args()
    
    violations = run_pyfmt(args.files)
    
    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
