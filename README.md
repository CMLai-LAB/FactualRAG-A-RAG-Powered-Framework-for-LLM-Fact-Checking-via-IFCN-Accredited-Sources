# FactualRAG-A-RAG-Powered-Framework-for-LLM-Fact-Checking-via-IFCN-Accredited-Sources

## 📋 Features

- **Complete Framework (`fact_checker.py`)**: Production-ready fact-checking system with vector database caching
- **RAG-based Verification (`fact_check_with_rag.py`)**: Evidence retrieval and analysis using RAG
- **Baseline Method (`fact_check_no_rag.py`)**: Direct fact-checking without external retrieval
- **Multi-Agent Debate (`fact_check_with_debate.py`)**: Consensus-based verification using multiple LLM agents

## 🚀 Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Google Custom Search API credentials (for web search)

### Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f requirements.yaml
conda activate fact-check
```

### Required Python Packages

- `langchain`
- `langchain-community`
- `langchain-ollama`
- `langchain-chroma`
- `langchain-huggingface`
- `pydantic`
- `fake-useragent`
- `requests`
- `fp` (FreeProxy)

## ⚙️ Configuration

Create an `APIKey.json` file in the root directory:

```json
{
  "search_api_key": "YOUR_GOOGLE_API_KEY",
  "search_engine_id": "YOUR_SEARCH_ENGINE_ID"
}
```

### Get Google Custom Search API Credentials

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Custom Search API
4. Create credentials (API Key)
5. Create a Custom Search Engine at [Programmable Search Engine](https://programmablesearchengine.google.com/)

## 📖 Usage

### 1. Complete Framework with Caching

Process multiple claims from a file:

```bash
python src/fact_checker.py \
  --input_path data/claims.json \
  --model llama3 \
  --output_path results/output.json
```

Process a single claim:

```bash
python src/fact_checker.py \
  --claim "The Earth is flat" \
  --urls "https://example.com/article1" "https://example.com/article2" \
  --model llama3 \
  --output_path results/single_claim.json
```

### 2. RAG-based Fact-Checking

```bash
python src/fact_check_with_rag.py \
  --input_path data/claims.json \
  --model gemma2 \
  --output_path results/rag_output.json
```

### 3. Baseline (No RAG)

```bash
python src/fact_check_no_rag.py \
  --input_path data/claims.json \
  --model phi4 \
  --output_path results/baseline_output.json
```

### 4. Multi-Agent Debate

```bash
python src/fact_check_with_debate.py \
  --input_path data/claims.json \
  --model llama3 \
  --output_path results/debate_output.json
```

## 📝 Input Format

Input JSON file should follow this structure:

```json
[
  {
    "claim": "Your claim text here",
    "urls": [
      "https://example.com/source1",
      "https://example.com/source2"
    ]
  }
]
```

## 📊 Output Format

All methods produce structured JSON output:

```json
{
  "claim": "Original claim text",
  "label": "Supported | Refuted | Not Enough Information",
  "language": "en",
  "date": "2024-01-15",
  "country": "US",
  "url": ["https://source1.com", "https://source2.com"],
  "reasoning": "Detailed reasoning process...",
  "time_taken": 12.34
}
```

### Label Definitions

- **Supported**: The claim is accurate and supported by evidence (True or Mostly True)
- **Refuted**: The claim is inaccurate and contradicted by evidence (False or Mostly False)
- **Not Enough Information**: Insufficient evidence or the claim is partially true (Half True)

## 🛠️ Supported Models

This framework supports any Ollama-compatible models:

- `llama3`
- `gemma2`
- `phi4:14b`
- Any other model available through Ollama

## 🔧 Advanced Features

### Vector Database Caching (`fact_checker.py`)

The complete framework includes automatic caching of previously analyzed claims:

- Uses Chroma vector database with HuggingFace embeddings
- Automatically retrieves cached results for similar claims
- Stores metadata (language, date, country, URLs)
- Significantly reduces processing time for repeated queries

### Multi-Agent Debate System

The debate system uses 5 independent agents over 5 rounds to:

1. Generate diverse perspectives
2. Challenge and refine reasoning
3. Reach consensus through iterative discussion
4. Produce more robust verdicts

## 📁 Project Structure

```
Fact-Check/
├── src/
│   ├── fact_checker.py              # Complete framework with caching
│   ├── fact_check_with_rag.py       # RAG-based method
│   ├── fact_check_no_rag.py         # Baseline method
│   └── fact_check_with_debate.py    # Multi-agent debate
├── scripts/
│   ├── run_all_models.sh            # Run all experiments
│   └── run_example.sh               # Example usage
├── results/
│   └── csv/                         # Experimental results
├── requirements.txt                  # Python dependencies
├── requirements.yaml                 # Conda environment
└── APIKey.json                       # API credentials (create this)
```

## 🔬 Running Experiments

Execute all experimental methods:

```bash
bash scripts/run_all_models.sh
```

Run a quick example:

```bash
bash scripts/run_example.sh
```

## 📊 Performance Considerations

- **Timeout Settings**: Default timeout is 10 seconds for web scraping
- **Retry Mechanism**: Automatic retry up to 20 times for failed LLM calls
- **Proxy Support**: Uses FreeProxy for anonymous web scraping
- **Memory Management**: Includes garbage collection to prevent memory leaks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

This research was conducted for academic purposes. If you use this code in your research, please cite our paper.

## ⚠️ Important Notes

1. **API Rate Limits**: Be aware of Google Custom Search API rate limits (100 queries/day for free tier)
2. **Proxy Usage**: FreeProxy may not always be reliable; consider using paid proxies for production
3. **LLM Costs**: While Ollama is free, ensure you have sufficient computational resources
4. **Ethical Use**: This tool is for research purposes. Always verify critical information manually.
