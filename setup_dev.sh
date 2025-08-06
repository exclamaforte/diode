#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up development environment for heur...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install the package in development mode with test dependencies
echo -e "${YELLOW}Installing package in development mode with test dependencies...${NC}"
pip install -e ".[test]"

# Run a simple test to verify the installation
echo -e "${YELLOW}Verifying installation...${NC}"
pytest -xvs tests/ --collect-only

echo -e "${GREEN}Development environment setup complete!${NC}"
echo -e "${GREEN}You can now run tests with: pytest${NC}"
echo -e "${GREEN}For code coverage: pytest --cov=heur${NC}"
echo -e "${GREEN}For linting: flake8 heur tests${NC}"
echo -e "${GREEN}For type checking: mypy heur${NC}"
echo -e "${YELLOW}Note: The virtual environment is now active. To deactivate, run: deactivate${NC}"
