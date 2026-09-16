"""
PyTorch / CUDA / CPU Inference Engine for Parallel Constrained Decoding.
Optimized for Linux containers, Hugging Face Spaces (ZeroGPU & CUDA), and cloud environments.
"""

import os
import time
import json
import copy
import re
import threading
from typing import Dict, Any, Generator, Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.cache_utils import DynamicCache

from core.schema import StructuredSchema
from core.prompt_builder import build_naive_json_prompt

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

_torch_model = None
_torch_tokenizer = None
_torch_device = None
_gpu_lock = threading.Lock()

# Support Hugging Face Spaces ZeroGPU if available
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=60)
except Exception:
    def gpu_decorator(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f


def get_torch_engine():
    global _torch_model, _torch_tokenizer, _torch_device
    if _torch_model is None or _torch_tokenizer is None:
        _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if _torch_device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        print(f"Loading {MODEL_ID} on {_torch_device} ({dtype})...")
        t0 = time.perf_counter()
        _torch_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True
        }
        if _torch_device == "cuda":
            load_kwargs["device_map"] = "auto"

        _torch_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        if _torch_device == "cpu":
            _torch_model = _torch_model.to("cpu")
        _torch_model.eval()
        print(f"Engine loaded on {_torch_device} in {time.perf_counter() - t0:.2f}s.")
        
    return _torch_model, _torch_tokenizer, _torch_device


def _resolve_collision_torch(model, batched_cache, base_len, row, suffix_len, cached_len,
                             field_logits, seqs, quote_id, temperature, device):
    """
    Torch counterpart of the MLX collision resolver.

    Choices that share a first token cannot be separated by a single-token score, so the
    decision is walked one token deeper at a time, constrained at each step to tokens that
    keep at least one declared choice reachable, until one choice survives. A closing quote
    terminates each choice so a choice that is a strict prefix of another stays separable.

    Returns (winner_idx, winner_prob, all_probs, extra_passes).
    """
    true_len = base_len + suffix_len
    f_cache = copy.deepcopy(batched_cache)
    f_cache.batch_select_indices(torch.tensor([row], device=device))
    # Drop this row's batch padding so the walk continues from the field's own suffix.
    overshoot = cached_len - true_len
    if overshoot > 0:
        f_cache.crop(-overshoot)

    alive = list(range(len(seqs)))
    cur_logits = field_logits
    path_prob = 1.0
    step = 0
    extra_passes = 0

    # Choices are unique (enforced in FieldDefinition), so the walk always separates
    # them within the longest choice's length; the cap is a guard against a future
    # change letting two choices tokenize identically and spinning here forever.
    max_steps = max((len(s) for s in seqs), default=0) + 1

    while len(alive) > 1 and step < max_steps:
        branches = {}
        for ci in alive:
            nxt = seqs[ci][step] if step < len(seqs[ci]) else quote_id
            branches.setdefault(nxt, []).append(ci)

        if len(branches) > 1:
            tok_ids = list(branches)
            scores = torch.tensor(
                [float(cur_logits[t].item()) for t in tok_ids], dtype=torch.float32
            ) / max(temperature, 1e-4)
            probs = F.softmax(scores, dim=-1)
            k = int(torch.argmax(probs).item())
            path_prob *= float(probs[k].item())
            chosen = tok_ids[k]
        else:
            chosen = next(iter(branches))

        alive = branches[chosen]
        step += 1
        if len(alive) > 1:
            with torch.no_grad():
                out_step = model(
                    torch.tensor([[chosen]], dtype=torch.long, device=device),
                    past_key_values=f_cache,
                )
            cur_logits = out_step.logits[0, -1, :]
            extra_passes += 1

    w_idx = alive[0]
    spread = (1.0 - path_prob) / max(len(seqs) - 1, 1)
    all_probs = [spread] * len(seqs)
    all_probs[w_idx] = path_prob
    return w_idx, path_prob, all_probs, extra_passes


@gpu_decorator
def run_parallel_generation_torch(
    context: str,
    schema: StructuredSchema,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Parallel Constrained Decision Engine running on PyTorch (CUDA / CPU).
    Evaluates all schema fields concurrently against a broadcast prefix KV-cache.
    """
    model, tokenizer, device = get_torch_engine()
    t0 = time.perf_counter()

    # 1. Compile schema metadata
    meta = schema.compile_parallel_metadata(tokenizer)
    field_items = meta["field_items"]
    suffix_lengths = meta["suffix_lengths"]
    cands_per_field = meta["cands_per_field"]
    prefixes = meta["prefixes"]
    has_collisions = meta["has_collisions"]
    suffixes_batch = meta["suffixes_batch"]
    M = len(field_items)

    # 2. High-density semantic catalog prefill
    schema_str = schema.to_parallel_schema_str()
    base_prompt = (
        f"<|im_start|>system\n"
        f"Classify JSON attributes:\n{schema_str}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n"
    )
    base_toks = tokenizer.encode(base_prompt, return_tensors="pt").to(device)

    t_pre0 = time.perf_counter()
    with torch.no_grad():
        base_out = model(base_toks, use_cache=True)
        base_cache = base_out.past_key_values
    t_prefill = (time.perf_counter() - t_pre0) * 1000

    # 3. Parallel Suffix Evaluation
    t_suf0 = time.perf_counter()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    suffix_arr = torch.tensor(suffixes_batch, dtype=torch.long, device=device)
    suffix_mask = (suffix_arr != pad_id).long()

    # Broadcast KV cache to batch size M
    with torch.no_grad():
        batched_cache = copy.deepcopy(base_cache)
        if hasattr(batched_cache, "batch_repeat_interleave"):
            batched_cache.batch_repeat_interleave(M)
        elif isinstance(batched_cache, tuple):
            batched_cache = tuple(
                tuple(t.repeat(M, 1, 1, 1) for t in layer)
                for layer in batched_cache
            )

        prefix_len = base_toks.shape[1]
        prefix_mask = torch.ones((M, prefix_len), dtype=torch.long, device=device)
        full_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        out = model(suffix_arr, past_key_values=batched_cache, attention_mask=full_mask)
        suffix_out = out.logits

    t_suffix_eval = (time.perf_counter() - t_suf0) * 1000

    # 4. Slicing, Disambiguation & Softmax
    parsed_json = {}
    field_telemetry = {}
    collision_passes = 0

    for i, (fname, fdef) in enumerate(field_items):
        decision_idx = suffix_lengths[i] - 1
        field_logits = suffix_out[i, decision_idx, :]
        cand_tokens = cands_per_field[i]

        if has_collisions[i]:
            w_idx, w_prob, all_probs, extra = _resolve_collision_torch(
                model, batched_cache, prefix_len, i, suffix_lengths[i],
                prefix_len + suffix_arr.shape[1], field_logits,
                meta["choice_seqs"][i], meta["quote_id"], temperature, device
            )
            collision_passes += extra
        else:
            scores = [float(field_logits[tid].item()) for tid in cand_tokens]
            scores_t = torch.tensor(scores, dtype=torch.float32) / max(temperature, 1e-4)
            probs = F.softmax(scores_t, dim=-1).tolist()
            w_idx = int(torch.argmax(scores_t).item())
            w_prob = float(probs[w_idx])
            all_probs = probs

        if fdef.field_type == "boolean":
            val = (w_idx == 0)
        else:
            val = fdef.choices[w_idx]

        parsed_json[fname] = {
            "value": val,
            "prob": round(w_prob, 4)
        }

        choices_list = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        scored_choices = []
        for c, p in zip(choices_list, all_probs):
            scored_choices.append({"choice": c, "probability": round(p, 4)})
        scored_choices.sort(key=lambda x: x["probability"], reverse=True)

        field_telemetry[fname] = {
            "value": val,
            "type": fdef.field_type,
            "confidence": round(w_prob, 4),
            "cardinality": fdef.cardinality,
            "top_choices": scored_choices[:5]
        }

    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "mode": "parallel_constrained_calibrated",
        "elapsed_ms": round(total_elapsed_ms, 2),
        "prefill_ms": round(t_prefill, 2),
        "suffix_eval_ms": round(t_suffix_eval, 2),
        "total_tokens_generated": 0,
        "sequential_forward_passes": 1 + collision_passes,
        "is_valid_json": True,
        "schema_match": True,
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "has_calibrated_probabilities": True,
        "num_fields": len(schema),
        "device": device
    }


@gpu_decorator
def run_naive_generation_torch(
    context: str,
    schema: StructuredSchema,
    temperature: float = 0.2,
    max_new_tokens: int = 512
) -> Dict[str, Any]:
    """
    Standard autoregressive baseline using PyTorch.
    """
    model, tokenizer, device = get_torch_engine()
    t0 = time.perf_counter()

    prompt = build_naive_json_prompt(context, schema)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_tokens = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-4),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    gen_tokens = output_ids.shape[1] - prompt_tokens
    tok_per_sec = (gen_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

    raw_text = tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)

    # Parse JSON
    parsed_json = None
    is_valid = False
    try:
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = raw_text[first_brace:last_brace + 1]
            parsed_json = json.loads(cleaned)
            is_valid = True
    except Exception:
        pass

    schema_match = False
    if is_valid and isinstance(parsed_json, dict):
        expected_keys = set(schema.get_field_names())
        schema_match = (set(parsed_json.keys()) == expected_keys)

    return {
        "mode": "autoregressive_naive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": gen_tokens,
        "tokens_per_second": round(tok_per_sec, 1),
        "sequential_forward_passes": gen_tokens,
        "is_valid_json": is_valid,
        "schema_match": schema_match,
        "raw_text": raw_text,
        "parsed_json": parsed_json,
        "device": device
    }


def stream_naive_generation_torch(
    context: str,
    schema: StructuredSchema,
    temperature: float = 0.2,
    max_new_tokens: int = 512
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator streaming individual tokens for side-by-side comparison visualizer.
    """
    model, tokenizer, device = get_torch_engine()
    t0 = time.perf_counter()

    prompt = build_naive_json_prompt(context, schema)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_tokens = input_ids.shape[1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": (temperature > 0.0),
        "temperature": max(temperature, 1e-4),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "streamer": streamer
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    full_text = ""
    tok_count = 0

    for token_str in streamer:
        tok_count += 1
        full_text += token_str
        yield {
            "type": "token",
            "token": token_str,
            "token_count": tok_count
        }

    thread.join()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    parsed_json = None
    is_valid = False
    try:
        first_brace = full_text.find("{")
        last_brace = full_text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = full_text[first_brace:last_brace + 1]
            parsed_json = json.loads(cleaned)
            is_valid = True
    except Exception:
        pass

    schema_match = False
    if is_valid and isinstance(parsed_json, dict):
        expected_keys = set(schema.get_field_names())
        schema_match = (set(parsed_json.keys()) == expected_keys)

    result = {
        "mode": "autoregressive_naive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": tok_count,
        "tokens_per_second": round((tok_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0, 1),
        "sequential_forward_passes": tok_count,
        "is_valid_json": is_valid,
        "schema_match": schema_match,
        "raw_text": full_text,
        "parsed_json": parsed_json,
        "device": device
    }

    yield {
        "type": "done",
        "result": result
    }
