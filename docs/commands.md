# Useful Commands & Shortcuts

Quick reference guide for common commands used throughout the course.

## 📦 Environment Management

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Remove virtual environment
rm -rf venv
```

### Conda Environment

```bash
# Create environment
conda create -n llm-course python=3.11

# Activate
conda activate llm-course

# Deactivate
conda deactivate

# List environments
conda env list

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# Remove environment
conda remove -n llm-course --all
```

## 📚 Package Management

### pip

```bash
# Install requirements
pip install -r requirements.txt

# Install with editable mode
pip install -e .

# Install specific package
pip install transformers==4.35.0

# Upgrade package
pip install --upgrade transformers

# Uninstall package
pip uninstall transformers

# Freeze requirements
pip freeze > requirements.txt

# Show package info
pip show transformers

# List installed packages
pip list
```

### pip-tools (Recommended)

```bash
# Install pip-tools
pip install pip-tools

# Compile requirements
pip-compile requirements.in

# Sync environment
pip-sync requirements.txt

# Upgrade all packages
pip-compile --upgrade requirements.in
```

## 🚀 Jupyter Commands

### Jupyter Lab/Notebook

```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook

# Start on specific port
jupyter lab --port=8889

# List running servers
jupyter lab list

# Stop server
jupyter lab stop 8888

# Generate config
jupyter lab --generate-config
```

### Jupyter Notebook Extensions

```bash
# Install extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable extension
jupyter nbextension enable <extension-name>

# List extensions
jupyter nbextension list
```

### Convert Notebooks

```bash
# Convert to Python script
jupyter nbconvert --to script notebook.ipynb

# Convert to HTML
jupyter nbconvert --to html notebook.ipynb

# Convert to PDF
jupyter nbconvert --to pdf notebook.ipynb

# Execute and convert
jupyter nbconvert --to html --execute notebook.ipynb
```

## 🤗 Hugging Face CLI

### Model Management

```bash
# Login
huggingface-cli login

# Download model
huggingface-cli download meta-llama/Llama-2-7b-hf

# Upload model
huggingface-cli upload my-username/my-model ./model

# List cached models
huggingface-cli scan-cache

# Delete cache
huggingface-cli delete-cache

# Model info
huggingface-cli info meta-llama/Llama-2-7b-hf
```

## 🔥 PyTorch Commands

### GPU Management

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# List GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Get GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### NVIDIA Commands

```bash
# Check NVIDIA driver
nvidia-smi

# Watch GPU usage (updates every 1s)
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -L

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Kill process using GPU
sudo fuser -v /dev/nvidia*
kill -9 <PID>
```

## 🗄️ Database Commands (for RAG)

### ChromaDB

```bash
# Install
pip install chromadb

# Start server
chroma run --path ./chroma_data

# Reset database
rm -rf ./chroma_data
```

### Weaviate

```bash
# Start with Docker
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:latest

# Stop container
docker stop <container-id>
```

## 🐳 Docker Commands

### Basic Commands

```bash
# Build image
docker build -t llm-course:latest .

# Run container
docker run -it --gpus all llm-course:latest

# Run with volume mount
docker run -it -v $(pwd):/workspace llm-course:latest

# List containers
docker ps -a

# Stop container
docker stop <container-id>

# Remove container
docker rm <container-id>

# Remove image
docker rmi llm-course:latest

# Clean up
docker system prune -a
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build
```

## 📊 Weights & Biases

```bash
# Login
wandb login

# Initialize project
wandb init

# Sync offline runs
wandb sync

# Pull artifacts
wandb artifact get <artifact-path>

# List projects
wandb projects
```

## 🧪 Testing Commands

### pytest

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose
pytest -v

# Run parallel
pytest -n auto

# Run specific test
pytest tests/test_model.py::test_function
```

### Testing with markers

```bash
# Run only fast tests
pytest -m fast

# Skip slow tests
pytest -m "not slow"

# Run GPU tests
pytest -m gpu
```

## 🔍 Code Quality

### Black (Formatting)

```bash
# Format file
black file.py

# Format directory
black src/

# Check without modifying
black --check src/

# Show diff
black --diff src/
```

### Ruff (Linting)

```bash
# Lint code
ruff check src/

# Fix auto-fixable issues
ruff check --fix src/

# Show rules
ruff rule S

# Check specific file
ruff check file.py
```

### isort (Import Sorting)

```bash
# Sort imports
isort file.py

# Sort directory
isort src/

# Check only
isort --check-only src/

# Show diff
isort --diff src/
```

### mypy (Type Checking)

```bash
# Type check
mypy src/

# Strict mode
mypy --strict src/

# Ignore missing imports
mypy --ignore-missing-imports src/
```

## 📈 Model Training

### Training Scripts

```bash
# Basic training
python train.py --config configs/bert.yaml

# Resume from checkpoint
python train.py --resume checkpoints/model_epoch_5.pt

# Distributed training (single node)
torchrun --nproc_per_node=2 train.py

# Distributed training (multi-node)
torchrun --nproc_per_node=2 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.1" \
         --master_port=1234 \
         train.py
```

### Accelerate

```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch train.py

# Multi-GPU training
accelerate launch --multi_gpu train.py

# DeepSpeed
accelerate launch --config_file ds_config.yaml train.py
```

## 🚀 Inference & Deployment

### vLLM

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000

# Start with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4
```

### Ollama

```bash
# Pull model
ollama pull llama2

# Run model
ollama run llama2

# List models
ollama list

# Show model info
ollama show llama2

# Remove model
ollama rm llama2

# Serve API
ollama serve
```

### TGI (Text Generation Inference)

```bash
# Run with Docker
docker run --gpus all \
  -p 8080:80 \
  -v $(pwd)/models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-hf
```

## 🔧 Git Commands

### Basic Workflow

```bash
# Clone repository
git clone <url>

# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "message"

# Push
git push origin main

# Pull
git pull origin main

# Create branch
git checkout -b feature/new-feature

# Switch branch
git checkout main

# Merge branch
git merge feature/new-feature

# Delete branch
git branch -d feature/new-feature
```

### Advanced Git

```bash
# Interactive rebase
git rebase -i HEAD~3

# Cherry-pick commit
git cherry-pick <commit-hash>

# Stash changes
git stash
git stash pop

# View history
git log --oneline --graph

# Reset to commit
git reset --hard <commit-hash>

# Clean untracked files
git clean -fd
```

## 📝 Quick Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Jupyter
alias jlab='jupyter lab'
alias jnb='jupyter notebook'

# Python
alias py='python'
alias ipy='ipython'

# Git
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'

# Docker
alias dps='docker ps'
alias dimg='docker images'

# PyTorch
alias gpu='watch -n 1 nvidia-smi'
```

---

**💡 Tip:** Create a `Makefile` to automate common tasks!

```makefile
install:
	pip install -r requirements.txt

test:
	pytest tests/

format:
	black src/ tests/
	isort src/ tests/

lint:
	ruff check src/ tests/
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
```
