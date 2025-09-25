#!/usr/bin/env python3
"""
Grep-based linter adapter for lintrunner.
"""

import argparse
import json
import re
from typing import Any, Dict, List, Optional


def run_grep_linter(
    files: List[str],
    pattern: str,
    linter_name: str,
    error_name: str,
    error_description: str,
    replace_pattern: Optional[str] = None,
    allowlist_pattern: Optional[str] = None,
    match_first_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run grep-based linting on the given files."""
    if not files:
        return []

    violations = []

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue

        for line_num, line in enumerate(lines, 1):
            # Check if line matches the pattern
            if re.search(pattern, line):
                # Check allowlist pattern if provided
                if allowlist_pattern and re.search(allowlist_pattern, line):
                    continue

                # Generate replacement if replace_pattern is provided
                replacement = None
                if replace_pattern:
                    try:
                        # Handle sed-style replacements
                        if replace_pattern.startswith("s/"):
                            parts = replace_pattern.split("/")
                            if len(parts) >= 3:
                                search_pat = parts[1]
                                replace_str = parts[2]
                                replacement = re.sub(
                                    search_pat, replace_str, line.rstrip("\n")
                                )
                    except re.error:
                        pass

                violations.append(
                    {
                        "path": file_path,
                        "line": line_num,
                        "char": 1,
                        "code": linter_name,
                        "severity": "error",
                        "name": linter_name,
                        "original": line.rstrip("\n") if replacement else None,
                        "replacement": replacement,
                        "description": error_description,
                    }
                )

                if match_first_only:
                    break

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grep-based linter")
    parser.add_argument("--pattern", required=True, help="Regex pattern to search for")
    parser.add_argument("--linter-name", required=True, help="Name of the linter")
    parser.add_argument("--error-name", required=True, help="Name of the error")
    parser.add_argument(
        "--error-description", required=True, help="Description of the error"
    )
    parser.add_argument("--replace-pattern", help="Replacement pattern (sed-style)")
    parser.add_argument(
        "--allowlist-pattern", help="Pattern to allowlist (ignore matches)"
    )
    parser.add_argument(
        "--match-first-only",
        action="store_true",
        help="Only match first occurrence per file",
    )
    parser.add_argument("files", nargs="*", help="Files to lint")

    args = parser.parse_args()

    violations = run_grep_linter(
        args.files,
        args.pattern,
        args.linter_name,
        args.error_name,
        args.error_description,
        args.replace_pattern,
        args.allowlist_pattern,
        args.match_first_only,
    )

    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
