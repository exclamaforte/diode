#!/usr/bin/env python3
"""
Pip initialization adapter for lintrunner.
"""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize pip packages for linting")
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="Don't actually install packages (0=install, 1=dry-run)",
    )
    parser.add_argument(
        "--no-black-binary",
        action="store_true",
        help="Don't use binary wheels for black",
    )
    parser.add_argument("packages", nargs="*", help="Packages to install")

    args = parser.parse_args()

    if args.dry_run:
        print(f"Would install packages: {args.packages}")
        return

    if not args.packages:
        return

    # Install packages
    cmd = [sys.executable, "-m", "pip", "install"]
    if args.no_black_binary:
        cmd.extend(["--no-binary", "black"])
    cmd.extend(args.packages)

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully installed: {args.packages}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
