#!/usr/bin/env python3
"""
Flake8 linter adapter for lintrunner.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


def run_flake8(files: List[str]) -> List[Dict[str, Any]]:
    """Run flake8 on the given files and return results."""
    if not files:
        return []

    # Filter out files that don't exist or are temporary files
    valid_files = []
    for file_path in files:
        if file_path.startswith("@/tmp/"):
            continue  # Skip temporary files that cause issues
        try:
            import os

            if os.path.exists(file_path):
                valid_files.append(file_path)
        except:
            continue

    if not valid_files:
        return []

    cmd = [
        sys.executable,
        "-m",
        "flake8",
        "--format=json",
        "--exit-zero",
    ] + valid_files

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            # Parse flake8 JSON output
            violations = json.loads(result.stdout)
            return [
                {
                    "path": violation["filename"],
                    "line": violation["line_number"],
                    "char": violation["column_number"],
                    "code": violation["code"],
                    "severity": "error"
                    if violation["code"].startswith("E")
                    else "warning",
                    "name": "FLAKE8",
                    "original": None,
                    "replacement": None,
                    "description": f"{violation['code']}: {violation['text']}",
                }
                for violation in violations
            ]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        # Fallback to basic format if JSON format fails
        cmd = [sys.executable, "-m", "flake8"] + valid_files
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        violations = []
        for line in result.stdout.splitlines():
            if ":" in line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    violations.append(
                        {
                            "path": parts[0],
                            "line": int(parts[1]) if parts[1].isdigit() else 1,
                            "char": int(parts[2]) if parts[2].isdigit() else 1,
                            "code": "FLAKE8",
                            "severity": "error",
                            "name": "FLAKE8",
                            "original": None,
                            "replacement": None,
                            "description": parts[3].strip(),
                        }
                    )
        return violations

    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run flake8 linter")
    parser.add_argument("files", nargs="*", help="Files to lint")

    args = parser.parse_args()

    violations = run_flake8(args.files)

    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
