#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up development environment for heur...${NC}"
echo -e "${YELLOW}This script will install packages in your current Python environment.${NC}"
echo -e "${YELLOW}If you're using a virtual environment or conda, make sure it's activated.${NC}"
read -p "Continue with installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Installation cancelled.${NC}"
    exit 1
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install the package in development mode with test and docs dependencies
echo -e "${YELLOW}Installing package in development mode with test and docs dependencies...${NC}"
pip install -e ".[test,docs]"

# Run a simple test to verify the installation
echo -e "${YELLOW}Verifying installation...${NC}"
pytest -xvs tests/ --collect-only

echo -e "${GREEN}Development environment setup complete!${NC}"
echo -e "${GREEN}You can now run tests with: pytest${NC}"
echo -e "${GREEN}For code coverage: pytest --cov=heur${NC}"
echo -e "${GREEN}For linting: flake8 heur tests${NC}"
echo -e "${GREEN}For type checking: mypy heur${NC}"
echo -e "${GREEN}To build documentation: cd docs && make html${NC}"
echo -e "${GREEN}To view documentation: open docs/_build/html/index.html${NC}"
echo -e "${GREEN}For live documentation preview: cd docs && make livehtml${NC}"
