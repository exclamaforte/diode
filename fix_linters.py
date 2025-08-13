#!/usr/bin/env python3
"""
Simple script to apply linter fixes to the diode project.

This provides easy commands for applying automatic fixes from linters.
"""

import subprocess
import sys


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Apply linter fixes using simple commands."""
    print("🚀 Diode Linter Fix Tool")
    print("=" * 50)
    
    # Try to run lintrunner on specific files to avoid temp file issues
    print("\n1️⃣  Applying formatting fixes to Python files...")
    if run_command(["lintrunner", "format", "diode/", "diode_common/", "examples/", "tests/"], "Apply formatting fixes to Python files"):
        print("\n💡 Formatting fixes applied! Checking for remaining issues...")
        if run_command(["lintrunner", "diode/", "diode_common/", "examples/", "tests/"], "Check remaining issues"):
            print("✅ All linting issues resolved!")
            return
        else:
            print("⚠️  Some non-formatting issues remain. Check output above.")
            return
    
    # Fallback: Try applying all patches to specific directories
    print("\n2️⃣  Trying all fixes on specific directories...")
    if run_command(["lintrunner", "--apply-patches", "diode/", "diode_common/", "examples/", "tests/"], "Apply all fixes to Python directories"):
        print("✅ All fixes applied successfully!")
        return
    
    # Option 3: Manual fix suggestions
    print("\n3️⃣  Manual fix options:")
    print("   • For formatting: lintrunner format diode/ diode_common/ examples/ tests/")
    print("   • For all fixes:  lintrunner --apply-patches diode/ diode_common/ examples/ tests/")
    print("   • Check issues:   lintrunner diode/ diode_common/ examples/ tests/")
    print("   • Use standalone: python run_linters.py --fix")
    
    sys.exit(1)


if __name__ == "__main__":
    main()
