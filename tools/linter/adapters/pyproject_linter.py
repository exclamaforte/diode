#!/usr/bin/env python3
"""
PyProject linter adapter for lintrunner.
"""

import argparse
import json
from typing import Any, Dict, List

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def run_pyproject_linter(files: List[str]) -> List[Dict[str, Any]]:
    """Check pyproject.toml files for basic validity."""
    if not files:
        return []
    
    if tomllib is None:
        return [{
            "path": "pyproject.toml",
            "line": 1,
            "char": 1,
            "code": "PYPROJECT",
            "severity": "error",
            "name": "PYPROJECT",
            "original": None,
            "replacement": None,
            "description": "tomllib/tomli not available for parsing TOML files",
        }]
    
    violations = []
    
    for file_path in files:
        if not file_path.endswith("pyproject.toml"):
            continue
        
        try:
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Basic validation checks
            if 'project' in data:
                project = data['project']
                
                # Check for required fields
                if 'name' not in project:
                    violations.append({
                        "path": file_path,
                        "line": 1,
                        "char": 1,
                        "code": "PYPROJECT",
                        "severity": "error",
                        "name": "PYPROJECT",
                        "original": None,
                        "replacement": None,
                        "description": "Missing required 'name' field in [project] section",
                    })
                
                if 'version' not in project:
                    violations.append({
                        "path": file_path,
                        "line": 1,
                        "char": 1,
                        "code": "PYPROJECT",
                        "severity": "warning",
                        "name": "PYPROJECT",
                        "original": None,
                        "replacement": None,
                        "description": "Missing 'version' field in [project] section",
                    })
        
        except (IOError, OSError) as e:
            violations.append({
                "path": file_path,
                "line": 1,
                "char": 1,
                "code": "PYPROJECT",
                "severity": "error",
                "name": "PYPROJECT",
                "original": None,
                "replacement": None,
                "description": f"Cannot read file: {e}",
            })
        except Exception as e:
            violations.append({
                "path": file_path,
                "line": 1,
                "char": 1,
                "code": "PYPROJECT",
                "severity": "error",
                "name": "PYPROJECT",
                "original": None,
                "replacement": None,
                "description": f"Invalid TOML syntax: {e}",
            })
    
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyproject.toml linter")
    parser.add_argument("files", nargs="*", help="Files to lint")
    
    args = parser.parse_args()
    
    violations = run_pyproject_linter(args.files)
    
    for violation in violations:
        print(json.dumps(violation))


if __name__ == "__main__":
    main()
