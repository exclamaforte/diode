#!/bin/bash

# Script to build the documentation for the diode project
# This builds the documentation from docs_src and outputs to docs

set -e

echo "Building diode documentation..."

# Store the base directory (parent of devops directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Navigate to the documentation source directory
cd "$BASE_DIR/docs_src"

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build HTML documentation
echo "Building HTML documentation..."
make html

cd ..
git add docs_src docs
git commit -m "Update documentation"
git push
