.PHONY: build-all build-torch-diode build-torch-diode-lib clean install-build-deps

# Build both packages
build-all: clean install-build-deps build-torch-diode build-torch-diode-lib

# Install build dependencies
install-build-deps:
	pip install build twine

# Clean previous builds
clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf /tmp/torch-diode-build /tmp/torch-diode-lib-build

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
	@cp -r examples/ /tmp/torch-diode-lib-build/
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
