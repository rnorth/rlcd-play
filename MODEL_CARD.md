---
language:
- en
license: apache-2.0
library_name: mlx
tags:
- structured-generation
- parallel-decoding
- constrained-decoding
- apple-silicon
- mlx
- classification
- json
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
---

# Qwen2.5-1.5B-Instruct with Parallel Constrained Decoding

This repository provides an inference implementation for structured JSON generation and high-cardinality classification using `mlx-community/Qwen2.5-1.5B-Instruct-4bit` on Apple Silicon.

Instead of generating structured JSON token-by-token through sequential autoregressive loops, this engine uses **Parallel Constrained Decoding**. It broadcasts the model KV-cache across all schema fields simultaneously, evaluating all decisions in parallel forward passes.

## Key Performance Highlights (Apple Silicon M4 Max)

- **High-Cardinality Decisions (255 choices)**: 89 ms total latency vs. 500 ms autoregressive baseline (5.6x faster).
- **Enterprise Multi-Field Extraction (28 fields)**: 270 ms total latency vs. 1,900 ms autoregressive baseline (7.0x faster).
- **Guaranteed Schema Validity**: 100% valid JSON syntax without grammar parsers, rejection sampling, or repair loops.
- **Calibrated Field Confidence**: Exact softmax probabilities computed directly over candidate token logits for every field.
- **Unified Memory Footprint**: ~1.1 GB total RAM footprint in 4-bit quantization on Apple Silicon.

## How It Works

Traditional structured output engines run standard autoregressive decoding. For an N-field JSON schema, the model performs hundreds of sequential forward passes:

```
[System + Prompt] -> Token 1 -> Token 2 -> ... -> Token K (O(N) sequential forward passes)
```

Parallel Constrained Decoding decomposes the structured generation task into an isolated broadcast pass:

1. **Prefix Prefill**: The context and semantic schema instructions are prefilled once. The resulting Key-Value (KV) cache is held in Apple Silicon Unified Memory.
2. **KV-Cache Broadcasting**: The KV-cache is broadcast across all target fields concurrently.
3. **Sub-Vocabulary Projection**: For each field, only valid candidate choices (e.g. enum options or boolean states) are evaluated. Unrelated vocabulary tokens are masked out.
4. **Calibrated Softmax**: Probabilities are computed directly via softmax over the candidate logit slice:
   $$P(c_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
5. **Collision Disambiguation**: In cases where candidate tokens share prefix strings, the engine follows continuation slices with zero memory reallocation.
6. **Programmatic Assembly**: The verified field choices and confidence scores are formatted directly into structured JSON.

## Quickstart SDK

### Installation

```bash
pip install -r requirements.txt
```

### Python Usage

```python
from core.schema import StructuredSchema
from core.engine import run_parallel_generation

# 1. Define schema
schema_definition = {
    "fraud_risk": {
        "type": "enum",
        "choices": ["LOW", "ELEVATED", "SUSPICIOUS", "CRITICAL"],
        "description": "Risk assessment tier for incoming transaction"
    },
    "block_account": {
        "type": "boolean",
        "description": "Whether immediate account restriction is required"
    },
    "recommended_action": {
        "type": "enum",
        "choices": ["ALLOW", "STEP_UP_2FA", "TEMPORARY_HOLD", "TERMINATE_SESSION"],
        "description": "Immediate mitigation action"
    }
}

schema = StructuredSchema(schema_definition)

# 2. Provide context
context = """
User ID: usr_9921
Location: Lagos, Nigeria (usual: Seattle, USA)
Device: Unknown Linux Chromium browser
Action: Wire transfer $49,500 to offshore escrow
Prior velocity: 0 transfers in 90 days
"""

# 3. Execute parallel generation
result = run_parallel_generation(context, schema)

print(f"Elapsed Time: {result['elapsed_ms']} ms")
print(f"Sequential Passes: {result['sequential_forward_passes']}")
print(f"Parsed JSON: {result['parsed_json']}")
```

### Output Example

```json
{
  "fraud_risk": { "value": "CRITICAL", "prob": 0.9942 },
  "block_account": { "value": "true", "prob": 0.9881 },
  "recommended_action": { "value": "TEMPORARY_HOLD", "prob": 0.9715 }
}
```

## Model Details

- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Quantization**: 4-bit AWQ (mlx-community format)
- **Context Window**: 32,768 tokens
- **Hardware Target**: Apple Silicon (M1, M2, M3, M4 series with unified memory)
- **Supported Field Types**: Categorical Enums (up to 255 choices per field) and Booleans

## Benchmark Summary

Evaluated on Apple Silicon M4 Max (128GB Unified Memory, MLX 0.22+):

| Scenario | Schema Fields | Autoregressive (ms) | Parallel Constrained (ms) | Speedup | Valid Syntax |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fintech Fraud Routing | 4 fields | 420 ms | 75 ms | **5.6x** | 100% |
| Code Security Audit | 4 fields | 380 ms | 68 ms | **5.6x** | 100% |
| High-Cardinality Tariff | 1 field (255 choices) | 500 ms | 89 ms | **5.6x** | 100% |
| Support Triage Matrix | 28 fields | 1,900 ms | 270 ms | **7.0x** | 100% |

