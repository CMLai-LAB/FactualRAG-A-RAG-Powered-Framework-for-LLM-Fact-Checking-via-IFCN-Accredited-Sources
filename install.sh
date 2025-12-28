#!/bin/bash

# Quick installation script for Fact-Check Framework
# This script automates the installation process

set -e  # Exit on error

echo "=================================="
echo "Fact-Check Framework Installation"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Error: Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"

# Check if Python version is >= 3.8
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Error: Python 3.8 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Check for Ollama
echo ""
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 || echo "unknown")
    echo "✓ Found Ollama: $OLLAMA_VERSION"
else
    echo "⚠️  Warning: Ollama is not installed."
    echo "   Please install Ollama from: https://ollama.ai/"
    echo "   The framework will not work without Ollama."
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check for APIKey.json
echo ""
if [ -f "APIKey.json" ]; then
    echo "✓ APIKey.json found"
else
    echo "⚠️  APIKey.json not found"
    echo ""
    echo "Would you like to create APIKey.json now? (y/N)"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please enter your Google Custom Search API key:"
        read -r API_KEY
        echo "Please enter your Search Engine ID:"
        read -r SEARCH_ID
        
        cat > APIKey.json << EOF
{
  "search_api_key": "$API_KEY",
  "search_engine_id": "$SEARCH_ID"
}
EOF
        echo "✓ APIKey.json created"
    else
        echo "You can create APIKey.json manually later."
        echo "Template:"
        echo '{'
        echo '  "search_api_key": "YOUR_API_KEY",'
        echo '  "search_engine_id": "YOUR_SEARCH_ENGINE_ID"'
        echo '}'
    fi
fi

# Test installation
echo ""
echo "Testing installation..."
$PYTHON_CMD -c "import langchain; import langchain_ollama; import langchain_chroma; print('✓ All modules loaded successfully')" || {
    echo "❌ Error: Some modules failed to load"
    exit 1
}

echo ""
echo "=================================="
echo "✓ Installation completed successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Make sure Ollama is running:"
echo "   ollama serve"
echo ""
echo "3. Download required models:"
echo "   ollama pull llama3"
echo "   ollama pull gemma2"
echo "   ollama pull phi4:14b"
echo ""
echo "4. Run an example:"
echo "   python src/fact_check_no_rag.py --help"
echo ""
echo "For more information, see README.md and INSTALL.md"
echo ""
