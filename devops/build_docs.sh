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

# Create .nojekyll file for GitHub Pages compatibility
echo "Creating .nojekyll file for GitHub Pages..."
touch "$BASE_DIR/docs/.nojekyll"

echo "Documentation built successfully!"
echo "HTML files are available in the docs/ directory"
echo "Open docs/index.html in your browser to view the documentation"
echo ""
echo "For GitHub Pages deployment:"
echo "1. Commit and push the docs/ directory"
echo "2. Configure GitHub Pages to serve from the docs/ folder"
