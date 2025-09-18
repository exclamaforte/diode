.PHONY: build-all build-torch-diode build-torch-diode-lib clean install-build-deps test test-unit test-integration test-cov test-cov-unit test-cov-integration coverage-report coverage-html coverage-xml install-test-deps test-module

# Build both packages
build-all: clean install-build-deps build-torch-diode build-torch-diode-lib

# Install build dependencies
install-build-deps:
	pip install build twine

# Clean previous builds
clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf /tmp/torch-diode-build /tmp/torch-diode-lib-build

# Clean all generated files (builds + coverage)
clean-all: clean clean-coverage

# Build torch-diode (with auto-registration)
build-torch-diode:
	@echo "Building torch-diode package..."
	python -m build --config-setting pyproject=pyproject.toml

# Build torch-diode-lib (without auto-registration)
build-torch-diode-lib:
	@echo "Building torch-diode-lib package..."
	@mkdir -p /tmp/torch-diode-lib-build
	@cp -r diode/ /tmp/torch-diode-lib-build/
	@cp -r trained_models/ /tmp/torch-diode-lib-build/
	@cp -r workflows/ /tmp/torch-diode-lib-build/
	@cp README.md LICENSE /tmp/torch-diode-lib-build/
	@cp pyproject-lib.toml /tmp/torch-diode-lib-build/pyproject.toml
	@# Replace the __init__.py with lib version
	@cp diode/__init___lib.py /tmp/torch-diode-lib-build/diode/__init__.py
	@cd /tmp/torch-diode-lib-build && python -m build
	@cp /tmp/torch-diode-lib-build/dist/* dist/
	@rm -rf /tmp/torch-diode-lib-build

# Test installation of both packages
test-install:
	@echo "Testing torch-diode installation..."
	pip install dist/torch_diode-*.whl --force-reinstall
	python -c "import diode; print('torch-diode installed successfully')"
	pip uninstall torch-diode -y

	@echo "Testing torch-diode-lib installation..."
	pip install dist/torch_diode_lib-*.whl --force-reinstall
	python -c "import diode; print('torch-diode-lib installed successfully')"
	pip uninstall torch-diode-lib -y

# Upload to PyPI (test)
upload-test:
	twine upload --repository testpypi dist/*

# Upload to PyPI (production)
upload:
	twine upload dist/*

# List built packages
list-packages:
	@echo "Built packages:"
	@ls -la dist/

# Install test dependencies
install-test-deps:
	pip install -e ".[test]"

# Run all tests
test: install-test-deps
	pytest

# Run unit tests only (excluding integration tests)
test-unit: install-test-deps
	pytest -m "not integration"

# Run integration tests only
test-integration: install-test-deps
	pytest -m "integration"

# Run all tests with coverage
test-cov: install-test-deps
	pytest --cov=diode --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml

# Run unit tests with coverage
test-cov-unit: install-test-deps
	pytest -m "not integration" --cov=diode --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml

# Run integration tests with coverage
test-cov-integration: install-test-deps
	pytest -m "integration" --cov=diode --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml

# Generate coverage report (assuming .coverage file exists)
coverage-report:
	coverage report

# Generate HTML coverage report
coverage-html:
	coverage html

# Generate XML coverage report
coverage-xml:
	coverage xml

# Clean coverage files
clean-coverage:
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f .coverage
	rm -f .coverage.*

# Run tests with coverage and open HTML report
test-and-view: test-cov
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	elif command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	else \
		echo "HTML coverage report generated at htmlcov/index.html"; \
	fi

# Run coverage on specific module/directory
# Usage: make test-module MODULE=model
# Usage: make test-module MODULE=types
# Usage: make test-module MODULE=integration
test-module: install-test-deps
	@if [ -z "$(MODULE)" ]; then \
		echo "Usage: make test-module MODULE=<module_name>"; \
		echo "Examples:"; \
		echo "  make test-module MODULE=model"; \
		echo "  make test-module MODULE=types"; \
		echo "  make test-module MODULE=integration"; \
		exit 1; \
	fi
	pytest tests/$(MODULE)/ --cov=diode.$(MODULE) --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml -v
