#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building documentation for heur...${NC}"

# Check if sphinx is installed
if ! pip show sphinx > /dev/null; then
    echo -e "${YELLOW}Sphinx not found. Installing documentation dependencies...${NC}"
    echo -e "${YELLOW}This will install packages in your current Python environment.${NC}"
    echo -e "${YELLOW}If you're using a virtual environment or conda, make sure it's activated.${NC}"
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -e ".[docs]"
    else
        echo -e "${RED}Installation cancelled. Please install the documentation dependencies manually:${NC}"
        echo -e "${RED}pip install -e \".[docs]\"${NC}"
        exit 1
    fi
fi

# Build the documentation
echo -e "${YELLOW}Building HTML documentation...${NC}"
cd docs
make clean
make html

# Check if the build was successful
if [ -d "_build/html" ]; then
    echo -e "${GREEN}Documentation built successfully!${NC}"
    echo -e "${GREEN}You can view the documentation by opening:${NC}"
    echo -e "${GREEN}$(pwd)/_build/html/index.html${NC}"

    # Try to open the documentation if on a system with a GUI
    if command -v xdg-open &> /dev/null; then
        echo -e "${YELLOW}Attempting to open documentation in browser...${NC}"
        xdg-open _build/html/index.html
    elif command -v open &> /dev/null; then
        echo -e "${YELLOW}Attempting to open documentation in browser...${NC}"
        open _build/html/index.html
    fi
else
    echo -e "${RED}Documentation build failed!${NC}"
    exit 1
fi

echo -e "${YELLOW}To serve the documentation locally, run:${NC}"
echo -e "${YELLOW}cd docs && python -m http.server -d _build/html${NC}"
