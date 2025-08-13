#!/usr/bin/env python3
"""
Codespell linter adapter for lintrunner.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


def run_codespell(files: List[str]) -> List[Dict[str, Any]]:
    """Run codespell on the given files and return results."""
    if not files:
        return []
    
    cmd = [sys.executable, "-m", "codespell", "--check-filenames"] + files
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        violations = []
        
        for line in result.stdout.splitlines():
            if ":" in line:
                # Parse codespell output format: file:line: word ==> suggestion
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    path = parts[0]
                    line_num = parts[1]
                    message = parts[2].strip()
                    
                    violations.append({
                        "path": path,
                        "line": int(line_num) if line_num.isdigit() else 1,
                        "char": 1,
                        "code": "CODESPELL",
                        "severity": "error",
                        "name": "CODESPELL",
                        "original": None,
                        "replacement": None,
                        "description": f"Spelling error: {message}",
                    })
        
        return violations
    except subprocess.CalledProcessError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run codespell linter")
    parser.add_argument("files", nargs="*", help="Files to lint")
    
    args = parser.parse_args()
    
    violations = run_codespell(args.files)
    
    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
