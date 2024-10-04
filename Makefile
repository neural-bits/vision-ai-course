PROJECT_NAME = edge-cv
PYTHON_VERSION = 3.10
CONDA_ENV_NAME = $(PROJECT_NAME)_env

define PRINT_LINE
	@echo "==================================================="
endef

define PRINT_STEP
	@echo "Step: $(1)"
endef

install_env:
	@echo "$(GREEN) [CONDA] Creating [$(ENV_NAME)] python env $(RESET)"
	conda create --name $(ENV_NAME) python=3.9 -y
	@echo "Activating the environment..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) \
	&& pip install poetry \
	poetry env use $(which python)"
	@echo "Installing Packages"
	@echo "Changing to pyproject.toml location..."
	@bash -c " PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring poetry install"

install:
	$(call PRINT_LINE)
	$(call PRINT_STEP, "Creating Conda environment with Python $(PYTHON_VERSION)...")
	conda create -y -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)
	$(call PRINT_LINE)
	$(call PRINT_STEP, "Activating environment and installing Poetry...")
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV_NAME) && pip install poetry && poetry env use $$(which python)"
	$(call PRINT_LINE)
	$(call PRINT_STEP, "Installing project dependencies using Poetry...")
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV_NAME) && PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring poetry install"
	$(call PRINT_LINE)
	$(call PRINT_STEP, "Installation complete!")
	$(call PRINT_LINE)


# Remove the Conda environment
clean:
	$(call PRINT_LINE)
	$(call PRINT_STEP, "Removing Conda environment...")
	conda remove -y --name $(CONDA_ENV_NAME) --all
	$(call PRINT_LINE)

# Help command to display all available targets
help:
	$(call PRINT_LINE)
	@echo "Available commands:"
	@echo "  make install       Create Conda environment and install dependencies"
	@echo "  make clean         Remove the Conda environment"
	@echo "  make help          Show this help message"
	$(call PRINT_LINE)