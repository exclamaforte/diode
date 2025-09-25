#!/usr/bin/env python3
"""
MyPy linter adapter for lintrunner.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


def run_mypy(
    files: List[str], config: str | None = None, code: str = "MYPY"
) -> List[Dict[str, Any]]:
    """Run mypy on the given files and return results."""
    if not files:
        return []

    cmd = [sys.executable, "-m", "mypy", "--show-error-codes"]

    if config:
        cmd.extend(["--config-file", config])

    cmd.extend(files)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        violations = []

        for line in result.stdout.splitlines():
            if ":" in line and " error:" in line:
                # Parse mypy output format: file:line: error: message [code]
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    path = parts[0]
                    line_num = parts[1]
                    rest = parts[2].strip()

                    # Extract error code if present
                    error_code = code
                    if "[" in rest and "]" in rest:
                        start = rest.rfind("[")
                        end = rest.rfind("]")
                        if start < end:
                            error_code = rest[start + 1 : end]
                            rest = rest[:start].strip()

                    violations.append(
                        {
                            "path": path,
                            "line": int(line_num) if line_num.isdigit() else 1,
                            "char": 1,
                            "code": error_code,
                            "severity": "error",
                            "name": code,
                            "original": None,
                            "replacement": None,
                            "description": rest,
                        }
                    )

        return violations
    except subprocess.CalledProcessError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mypy linter")
    parser.add_argument("--config", help="Path to mypy config file")
    parser.add_argument("--code", default="MYPY", help="Linter code name")
    parser.add_argument("files", nargs="*", help="Files to lint")

    args = parser.parse_args()

    violations = run_mypy(args.files, args.config, args.code)

    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
