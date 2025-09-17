#!/bin/bash
# Simple wrapper script for PyPI deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy_to_pypi.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <test|prod> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  test        Deploy to TestPyPI"
    echo "  prod        Deploy to production PyPI"
    echo ""
    echo "Options:"
    echo "  --build-only    Only build packages, don't upload"
    echo "  --skip-build    Skip building, use existing dist files"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test              # Build and deploy to TestPyPI"
    echo "  $0 prod              # Build and deploy to production PyPI"
    echo "  $0 test --build-only # Only build packages for testing"
}

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

ENVIRONMENT="$1"
shift

if [[ "$ENVIRONMENT" != "test" && "$ENVIRONMENT" != "prod" ]]; then
    echo -e "${RED}Error: Environment must be 'test' or 'prod'${NC}" >&2
    print_usage
    exit 1
fi

# Check if we're in the diode directory
if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
    echo -e "${RED}Error: deploy_to_pypi.py not found. Make sure you're running this from the diode directory.${NC}" >&2
    exit 1
fi

# Check if Python script is executable
if [[ ! -x "$DEPLOY_SCRIPT" ]]; then
    echo -e "${YELLOW}Making deploy_to_pypi.py executable...${NC}"
    chmod +x "$DEPLOY_SCRIPT"
fi

# Run the Python deployment script
echo -e "${GREEN}Starting deployment to $ENVIRONMENT PyPI...${NC}"
python3 "$DEPLOY_SCRIPT" "$ENVIRONMENT" "$@"
