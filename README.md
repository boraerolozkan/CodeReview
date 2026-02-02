# CodeReview AI

An intelligent code review model that automatically detects bugs, security vulnerabilities, and code quality issues across multiple programming languages.

Built on **Qwen2.5-Coder-7B** and fine-tuned with **LoRA** for efficient, accurate code analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--Coder--7B-purple.svg)

> **Model Weights**: Download from [Hugging Face Hub](https://huggingface.co/boraoxkan/codereview-ai)

---

## Features

- **Multi-Language Support**: Python, JavaScript, Java, C++, Go, Rust, TypeScript, C#, SQL
- **Security Analysis**: Detects OWASP Top 10 vulnerabilities (SQL Injection, XSS, Command Injection, etc.)
- **Code Quality Scoring**: 0-100 quality score with detailed explanations
- **Auto-Fix Suggestions**: Provides corrected code snippets
- **Memory Efficient**: 4-bit quantization runs on consumer GPUs (8GB VRAM)
- **Fast Inference**: Optimized with Unsloth for 2x faster generation

---

## Demo Output

```
Input Code:
def get_user(username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)

Model Analysis:
{
  "code_quality_score": 20,
  "critical_issues": [
    "SQL Injection vulnerability due to direct string concatenation"
  ],
  "suggestions": [
    "Use parameterized queries to prevent SQL injection",
    "Handle database connections properly to avoid resource leaks"
  ],
  "fixed_code": "def get_user(username):\n    query = \"SELECT * FROM users WHERE username = ?\"\n    cursor.execute(query, (username,))"
}
```

---

## Project Structure

```
CodeReview/
├── models/
│   └── github_agent_v1/          # Fine-tuned model weights
│       ├── adapter_config.json   # LoRA configuration
│       ├── tokenizer.json        # Tokenizer vocabulary
│       └── ...
├── scripts/
│   ├── train.py                  # Model training script
│   ├── inference.py              # Basic inference example
│   ├── test_model.py             # Comprehensive test suite
│   ├── generator.py              # Synthetic data generation
│   ├── preprocessor.py           # Data cleaning pipeline
│   └── analytics.py              # Dataset analysis & visualization
├── dataset/
│   ├── raw/                      # Raw generated data
│   └── clean/                    # Preprocessed training data
├── analytics/
│   └── Figure_1.png              # Dataset distribution charts
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- CUDA Toolkit 11.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeReview.git
cd CodeReview

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install unsloth
pip install transformers datasets trl peft
pip install pandas matplotlib seaborn
```

---

## Usage

### Quick Test

```bash
cd scripts
python test_model.py --quick
```

### Full Test Suite

```bash
python test_model.py
```

Tests 8 different vulnerability types:
- SQL Injection (Python)
- XSS Vulnerability (JavaScript)
- Command Injection (Python)
- Hardcoded Credentials (Java)
- Race Condition (Python)
- Memory Leak (C++)
- Off-by-One Error (JavaScript)
- Null Pointer (Java)

### Interactive Mode

Analyze your own code interactively:

```bash
python test_model.py --interactive
```

### Basic Inference

```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/github_agent_v1",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Prepare prompt
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze this Python code for defects.

### Input:
def login(user, pwd):
    if user == "admin" and pwd == "password123":
        return True
    return False

### Response:
"""

# Generate analysis
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

---

## Model Details

### Architecture

| Component | Details |
|-----------|---------|
| Base Model | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` |
| Parameters | 7 Billion |
| Quantization | 4-bit (NF4) |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| Context Length | 2048 tokens |

### LoRA Configuration

```python
r = 16                    # Rank
lora_alpha = 16           # Alpha
lora_dropout = 0          # Dropout
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 1 (effective: 4 with gradient accumulation) |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Training Steps | 120 |
| Warmup Steps | 10 |
| Precision | BF16/FP16 |

---

## Dataset

### Overview

- **Total Samples**: ~500 code snippets with bugs
- **Format**: Instruction-Input-Output (Alpaca format)
- **Generation**: Synthetic data via Qwen2.5:32b as teacher model

### Bug Categories (25+ types)

**Security Vulnerabilities:**
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Hardcoded Credentials
- Path Traversal
- Insecure Deserialization

**Code Quality Issues:**
- Memory Leaks
- Race Conditions
- Null Pointer Dereference
- Off-by-One Errors
- Resource Leaks
- Infinite Loops

**Performance Problems:**
- N+1 Query Issues
- Inefficient Algorithms
- Unnecessary Allocations

### Language Distribution

| Language | Percentage |
|----------|------------|
| Python | ~20% |
| JavaScript | ~15% |
| Java | ~15% |
| C++ | ~12% |
| Go | ~12% |
| TypeScript | ~10% |
| Rust | ~8% |
| C# | ~5% |
| SQL | ~3% |

---

## Training

### Generate Training Data

```bash
cd scripts

# Generate synthetic buggy code samples (requires Ollama with qwen2.5:32b)
python generator.py

# Clean and preprocess the data
python preprocessor.py
```

### Train the Model

```bash
python train.py
```

Training takes approximately **30-45 minutes** on RTX 3070 (8GB VRAM).

### Monitor Training

Training logs show loss values per step:
```
Step 1: loss=2.345
Step 10: loss=1.234
Step 50: loss=0.567
Step 120: loss=0.234
```

---

## Output Format

The model outputs JSON with the following structure:

```json
{
  "code_quality_score": 0-100,
  "critical_issues": [
    "Description of critical bug or vulnerability"
  ],
  "suggestions": [
    "Recommendation for improvement"
  ],
  "fixed_code": "Corrected version of the code"
}
```

### Score Guidelines

| Score | Quality Level | Description |
|-------|--------------|-------------|
| 0-30 | Critical | Severe security vulnerabilities or critical bugs |
| 31-50 | Poor | Significant issues requiring immediate attention |
| 51-70 | Fair | Some issues present, improvements recommended |
| 71-85 | Good | Minor issues, generally acceptable |
| 86-100 | Excellent | Clean, secure, well-written code |

---

## Hardware Requirements

### Minimum (Inference)
- GPU: 6GB VRAM (RTX 2060 / RTX 3060)
- RAM: 16GB
- Storage: 10GB

### Recommended (Training)
- GPU: 8GB+ VRAM (RTX 3070 / RTX 4070)
- RAM: 32GB
- Storage: 20GB

---

## Limitations

- Context limited to 2048 tokens (large files may need splitting)
- Optimized for single-function analysis rather than entire codebases
- May produce false positives for complex, unconventional patterns
- Training data is synthetically generated

---

## Roadmap

- [ ] GGUF export for Ollama integration
- [ ] VS Code extension
- [ ] GitHub Actions integration
- [ ] Support for more languages (PHP, Ruby, Kotlin)
- [ ] Multi-file context analysis
- [ ] Fine-tuning on real-world vulnerability datasets

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning library
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) - Base model
- [Hugging Face](https://huggingface.co/) - Transformers & TRL libraries

---

## Citation

```bibtex
@software{codereview_ai,
  title = {CodeReview AI: Automated Code Analysis with Fine-tuned LLMs},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/CodeReview}
}
```

---

## Contact

- GitHub: [Bora Erol Özkan](https://github.com/boraerolozkan)
- Email: boraerolozkan@gmail.com

---

<p align="center">
  <b>Built with Unsloth & Qwen2.5-Coder</b><br>
  <sub>Making code reviews smarter, one bug at a time.</sub>
</p>
