#!/usr/bin/env python3
"""
Ruff linter adapter for lintrunner.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


def run_ruff(files: List[str], config: str | None = None, show_disable: bool = False, apply_fixes: bool = False) -> List[Dict[str, Any]]:
    """Run ruff on the given files and return results."""
    if not files:
        return []
    
    if apply_fixes:
        # Apply fixes using ruff check --fix
        cmd = [sys.executable, "-m", "ruff", "check", "--fix"]
        if config:
            cmd.extend(["--config", config])
        cmd.extend(files)
        
        try:
            subprocess.run(cmd, check=False)
            return []  # No violations to report if fixes were applied
        except subprocess.CalledProcessError:
            pass
    
    # Check for violations
    cmd = [sys.executable, "-m", "ruff", "check", "--output-format=json"]
    
    if config:
        cmd.extend(["--config", config])
    
    cmd.extend(files)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        violations = []
        
        if result.stdout:
            try:
                ruff_output = json.loads(result.stdout)
                for violation in ruff_output:
                    violations.append({
                        "path": violation["filename"],
                        "line": violation["location"]["row"],
                        "char": violation["location"]["column"],
                        "code": violation["code"],
                        "severity": "error" if violation["code"].startswith("E") else "warning",
                        "name": "RUFF",
                        "original": None,
                        "replacement": violation.get("fix", {}).get("content") if violation.get("fix") else None,
                        "description": f"{violation['code']}: {violation['message']}",
                    })
            except (json.JSONDecodeError, KeyError):
                # Fallback to parsing text output
                for line in result.stdout.splitlines():
                    if ":" in line:
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            violations.append({
                                "path": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 1,
                                "char": int(parts[2]) if parts[2].isdigit() else 1,
                                "code": "RUFF",
                                "severity": "error",
                                "name": "RUFF",
                                "original": None,
                                "replacement": None,
                                "description": parts[3].strip(),
                            })
        
        return violations
    except subprocess.CalledProcessError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ruff linter")
    parser.add_argument("--config", help="Path to ruff config file")
    parser.add_argument("--show-disable", action="store_true", help="Show disable information")
    parser.add_argument("files", nargs="*", help="Files to lint")
    
    args = parser.parse_args()
    
    violations = run_ruff(args.files, args.config, args.show_disable)
    
    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
