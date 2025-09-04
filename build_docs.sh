#!/bin/bash

# Script to build the documentation for the diode project
# This builds the documentation from docs-src and outputs to docs

set -e

echo "Building diode documentation..."

# Navigate to the documentation source directory
cd "$(dirname "$0")/docs-src"

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build HTML documentation  
echo "Building HTML documentation..."
make html

echo "Documentation built successfully!"
echo "HTML files are available in the docs/ directory"
echo "Open docs/index.html in your browser to view the documentation"