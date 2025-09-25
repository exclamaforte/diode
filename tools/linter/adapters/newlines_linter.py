#!/usr/bin/env python3
"""
Newlines linter adapter for lintrunner.
"""

import argparse
import json
from typing import Any, Dict, List


def run_newlines_linter(files: List[str]) -> List[Dict[str, Any]]:
    """Check files for proper newline endings."""
    if not files:
        return []

    violations = []

    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            if not content:
                continue

            # Check if file ends with newline
            if not content.endswith(b"\n"):
                violations.append(
                    {
                        "path": file_path,
                        "line": 1,
                        "char": 1,
                        "code": "NEWLINE",
                        "severity": "error",
                        "name": "NEWLINE",
                        "original": None,
                        "replacement": None,
                        "description": "File does not end with a newline",
                    }
                )

        except OSError:
            continue

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run newlines linter")
    parser.add_argument("files", nargs="*", help="Files to lint")

    args = parser.parse_args()

    violations = run_newlines_linter(args.files)

    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
