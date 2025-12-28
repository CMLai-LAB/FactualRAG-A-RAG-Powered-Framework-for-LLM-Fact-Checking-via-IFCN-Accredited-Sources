# Installation Guide

This guide provides detailed instructions for installing the Fact-Checking Framework.

## Prerequisites

### 1. Python Environment

- **Python 3.8 or higher** is required
- We recommend using **Python 3.10+** for best compatibility

Check your Python version:
```bash
python --version
# or
python3 --version
```

### 2. Ollama

This framework requires [Ollama](https://ollama.ai/) to run LLM models locally.

**Install Ollama:**

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

**Verify installation:**
```bash
ollama --version
```

**Download required models:**
```bash
ollama pull llama3
ollama pull gemma2
ollama pull phi4:14b
```

### 3. Google Custom Search API (Optional but Recommended)

If you want to use the automatic URL retrieval feature:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the **Custom Search API**
4. Create **API credentials** (API Key)
5. Set up a **Programmable Search Engine** at [programmablesearchengine.google.com](https://programmablesearchengine.google.com/)

## Installation Methods

### Method 1: Using pip (Recommended)

**Step 1: Clone the repository**
```bash
git clone https://github.com/CMLai-LAB/Fact-Check.git
cd Fact-Check
```

**Step 2: Create a virtual environment (recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Configure API keys**
```bash
# Create APIKey.json in the root directory
cat > APIKey.json << EOF
{
  "search_api_key": "YOUR_GOOGLE_API_KEY",
  "search_engine_id": "YOUR_SEARCH_ENGINE_ID"
}
EOF
```

### Method 2: Using Conda

**Step 1: Clone the repository**
```bash
git clone https://github.com/CMLai-LAB/Fact-Check.git
cd Fact-Check
```

**Step 2: Create conda environment**
```bash
conda env create -f requirements.yaml
```

**Step 3: Activate environment**
```bash
conda activate FactCheck
```

**Step 4: Configure API keys** (same as Method 1, Step 4)

### Method 3: Using setup.py (Development)

For development or if you want to install the package system-wide:

```bash
git clone https://github.com/CMLai-LAB/Fact-Check.git
cd Fact-Check
pip install -e .
```

The `-e` flag installs in editable mode, useful for development.

## Verification

Test your installation:

```bash
# Test imports
python -c "import langchain; import langchain_ollama; print('Installation successful!')"

# Check Ollama connection
ollama list

# Test a simple fact-check (requires Ollama to be running)
python src/fact_check_no_rag.py --help
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'fake_useragent'`

**Solution:**
```bash
pip install fake-useragent
```

### Issue: `Connection refused` when running models

**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

### Issue: Chroma database errors

**Solution:** Install chromadb explicitly:
```bash
pip install chromadb --upgrade
```

### Issue: SSL Certificate errors

**Solution:** Update certificates:
```bash
pip install --upgrade certifi
```

### Issue: Free-proxy not working

**Solution:** The free-proxy library may have reliability issues. Consider:
1. Using paid proxy services
2. Running without proxies (remove proxy from code)
3. Installing alternative: `pip install free-proxy==1.1.1`

### Issue: Memory errors with large models

**Solution:** 
- Use smaller models (e.g., `phi4` instead of `llama3`)
- Increase system swap space
- Reduce `num_predict` parameter in the code

## System Requirements

### Minimum Requirements:
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Storage:** 10 GB free space
- **OS:** Linux, macOS, or Windows 10+

### Recommended Requirements:
- **CPU:** 8+ cores
- **RAM:** 16+ GB
- **Storage:** 20+ GB free space (for models)
- **GPU:** Optional, but speeds up inference

## Optional: GPU Support

For faster inference with GPU:

**NVIDIA GPU (CUDA):**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPU (ROCm):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Apple Silicon (MPS):**
PyTorch with MPS support is included by default on macOS with Apple Silicon.

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage instructions
2. Try the example scripts in `scripts/`
3. Configure your datasets in the appropriate format
4. Run experiments with different models

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/CMLai-LAB/Fact-Check/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Updating

To update to the latest version:

```bash
cd Fact-Check
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

**If using virtual environment:**
```bash
deactivate
rm -rf venv
```

**If using conda:**
```bash
conda deactivate
conda env remove -n FactCheck
```

**If installed with setup.py:**
```bash
pip uninstall fact-check-framework
```
