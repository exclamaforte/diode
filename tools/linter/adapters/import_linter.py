#!/usr/bin/env python3
"""
Import linter adapter for lintrunner.
"""

import argparse
import ast
import json
from typing import Any, Dict, List

# Define disallowed imports for different contexts
DISALLOWED_IMPORTS = {
    "default": {
        # Add any globally disallowed imports here
    },
    "diode": {
        # Add diode-specific disallowed imports here
        # Example: "requests": "Use urllib instead of requests for HTTP calls"
    },
}


def check_imports_in_file(
    file_path: str, context: str = "default"
) -> List[Dict[str, Any]]:
    """Check for disallowed imports in a Python file."""
    violations = []
    disallowed = DISALLOWED_IMPORTS.get(context, {})

    if not disallowed:
        return violations

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            # Skip files with syntax errors
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]  # Get top-level module
                    if module_name in disallowed:
                        violations.append(
                            {
                                "path": file_path,
                                "line": node.lineno,
                                "char": node.col_offset,
                                "code": "IMPORT_LINTER",
                                "severity": "error",
                                "name": "IMPORT_LINTER",
                                "original": None,
                                "replacement": None,
                                "description": f"Disallowed import '{module_name}': {disallowed[module_name]}",
                            }
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]  # Get top-level module
                    if module_name in disallowed:
                        violations.append(
                            {
                                "path": file_path,
                                "line": node.lineno,
                                "char": node.col_offset,
                                "code": "IMPORT_LINTER",
                                "severity": "error",
                                "name": "IMPORT_LINTER",
                                "original": None,
                                "replacement": None,
                                "description": f"Disallowed import from '{module_name}': {disallowed[module_name]}",
                            }
                        )

    except OSError:
        pass

    return violations


def run_import_linter(files: List[str]) -> List[Dict[str, Any]]:
    """Run import linting on the given files."""
    if not files:
        return []

    violations = []

    for file_path in files:
        if not file_path.endswith(".py"):
            continue

        # Determine context based on file path
        context = "default"
        if "diode" in file_path:
            context = "diode"

        violations.extend(check_imports_in_file(file_path, context))

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run import linter")
    parser.add_argument("files", nargs="*", help="Files to lint")

    args = parser.parse_args()

    violations = run_import_linter(args.files)

    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
