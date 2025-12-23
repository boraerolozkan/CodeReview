---
language:
- en
license: mit
library_name: peft
tags:
- code-review
- code-analysis
- security
- bug-detection
- vulnerability-detection
- qwen2
- lora
- unsloth
- sft
- transformers
- trl
base_model: unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit
pipeline_tag: text-generation
datasets:
- custom
model-index:
- name: codereview-ai
  results: []
---

<div align="center">

# CodeReview AI

**Automated Code Review with Fine-tuned LLMs**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/boraoxkan/CodeReview)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--Coder--7B-purple)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

</div>

---

## Overview

A fine-tuned code review model that automatically detects **bugs**, **security vulnerabilities**, and **code quality issues** across multiple programming languages.

### Key Features

- **Multi-Language**: Python, JavaScript, Java, C++, Go, Rust, TypeScript, C#, SQL
- **Security Focus**: Detects OWASP Top 10 vulnerabilities
- **Quality Scoring**: 0-100 score with explanations
- **Auto-Fix**: Provides corrected code snippets
- **Efficient**: 4-bit quantization, runs on 8GB VRAM

---

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen2.5-Coder-7B-Instruct |
| **Parameters** | 7B |
| **Fine-tuning** | LoRA (r=16, alpha=16) |
| **Quantization** | 4-bit NF4 |
| **Context Length** | 2048 tokens |
| **Framework** | Unsloth + TRL |

---

## Detected Issues

<table>
<tr>
<td>

**Security**
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Hardcoded Credentials
- Path Traversal
- Insecure Deserialization

</td>
<td>

**Code Quality**
- Memory Leaks
- Race Conditions
- Null Pointer Dereference
- Off-by-One Errors
- Resource Leaks
- Infinite Loops

</td>
</tr>
</table>

---

## Quick Start

```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="boraoxkan/codereview-ai",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Analyze code
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze this Python code for defects.

### Input:
def get_user(username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()

### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
result = tokenizer.decode(outputs[0])
```

---

## Example Output

**Input Code (SQL Injection vulnerability):**
```python
def get_user(username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
```

**Model Output:**
```json
{
  "code_quality_score": 20,
  "critical_issues": [
    "SQL Injection vulnerability due to direct string concatenation"
  ],
  "suggestions": [
    "Use parameterized queries to prevent SQL injection",
    "Handle database connections properly"
  ],
  "fixed_code": "def get_user(username):\n    query = \"SELECT * FROM users WHERE username = ?\"\n    cursor.execute(query, (username,))"
}
```

---

## Score Guidelines

| Score | Level | Description |
|:-----:|:-----:|-------------|
| 0-30 | **Critical** | Severe security vulnerabilities |
| 31-50 | **Poor** | Significant issues present |
| 51-70 | **Fair** | Some improvements needed |
| 71-85 | **Good** | Minor issues only |
| 86-100 | **Excellent** | Clean, secure code |

---

## Training

| Parameter | Value |
|-----------|-------|
| Dataset | ~500 synthetic samples |
| Steps | 120 |
| Batch Size | 1 (effective: 4) |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Precision | BF16 |
| Hardware | RTX 3070 (8GB) |
| Time | ~40 minutes |

### LoRA Config

```python
r = 16
lora_alpha = 16
lora_dropout = 0
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

---

## Limitations

- Context limited to 2048 tokens
- Optimized for single-function analysis
- May produce false positives for complex patterns
- Training data is synthetically generated

---

## Links

| Resource | Link |
|----------|------|
| GitHub Repository | [boraoxkan/CodeReview](https://github.com/boraoxkan/CodeReview) |
| Base Model | [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| Unsloth | [unslothai/unsloth](https://github.com/unslothai/unsloth) |

---

## Citation

```bibtex
@software{codereview_ai_2025,
  title = {CodeReview AI: Automated Code Analysis with Fine-tuned LLMs},
  author = {Bora Ozkan},
  year = {2025},
  url = {https://huggingface.co/boraoxkan/codereview-ai}
}
```

---

## License

MIT License - See [LICENSE](https://github.com/boraoxkan/CodeReview/blob/main/LICENSE) for details.

---

<div align="center">
<b>Built with Unsloth & Qwen2.5-Coder</b><br>
<sub>Making code reviews smarter, one bug at a time.</sub>
</div>
