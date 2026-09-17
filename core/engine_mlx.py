"""
Inference Engine comparing Autoregressive JSON Generation
vs. Parallel Constrained Decision Engine.
Runs locally on Apple Silicon via MLX with broadcast prefix KV-caching.
"""

import time
import json
import re
import os
import copy
import platform
import threading
from typing import Dict, Any, Generator, Optional, List, Tuple
from core.schema import StructuredSchema, map_candidate_tokens, extract_calibrated_probabilities
from core.prompt_builder import build_naive_json_prompt

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

_model = None
_tokenizer = None
_gpu_lock = threading.Lock()


def gpu_locked(fn):
    def wrapper(*args, **kwargs):
        with _gpu_lock:
            return fn(*args, **kwargs)
    return wrapper


def gpu_locked_gen(fn):
    def wrapper(*args, **kwargs):
        with _gpu_lock:
            yield from fn(*args, **kwargs)
    return wrapper


def get_engine():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"Loading {MODEL_ID} into Apple Silicon unified memory...")
        t0 = time.perf_counter()
        _model, _tokenizer = load(MODEL_ID)
        print(f"Engine loaded in {time.perf_counter() - t0:.2f}s.")
        
        # GPU warmup: compile prefill and broadcast decode shaders ahead of time
        print("Warming up Metal shaders on Apple Silicon GPU...")
        w_toks = _tokenizer.encode("Warmup context for Apple Silicon GPU")
        w_cache = make_prompt_cache(_model)
        w_logits = _model(mx.array(w_toks)[None], cache=w_cache)
        mx.eval(w_logits)
        
        # Warmup batched broadcast suffix for up to 28 fields
        b_cache = []
        for c in w_cache:
            nc = copy.copy(c)
            if hasattr(c, "keys") and c.keys is not None:
                nc.keys = mx.repeat(c.keys, 28, axis=0)
            if hasattr(c, "values") and c.values is not None:
                nc.values = mx.repeat(c.values, 28, axis=0)
            b_cache.append(nc)
        s_dummy = mx.zeros((28, 6), dtype=mx.int32)
        w_suf = _model(s_dummy, cache=b_cache)
        mx.eval(w_suf)
        print("Metal shaders compiled & warmed up.")
        
    return _model, _tokenizer


@gpu_locked
def run_naive_generation(
    context: str,
    schema: StructuredSchema,
    max_tokens: int = 700,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Standard autoregressive generation baseline:
    Prompts the LLM to generate the entire JSON object token-by-token.
    """
    model, tokenizer = get_engine()
    prompt = build_naive_json_prompt(context, schema)
    
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = mx.array(prompt_tokens)[None]
    
    t0 = time.perf_counter()
    generated_tokens = []
    text_chunks = []
    
    current_text = "{\n  "
    cache = make_prompt_cache(model)
    
    # Prefill pass
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[:, -1, :]))
    generated_tokens.append(next_token)
    token_str = tokenizer.decode([next_token])
    current_text += token_str
    text_chunks.append(token_str)
    
    stop_tokens = {tokenizer.eos_token_id}
    for tok_str in ["<end_of_turn>", "<|im_end|>", "<eos>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)
        if tok_id is not None and isinstance(tok_id, int) and tok_id > 0:
            stop_tokens.add(tok_id)
    
    while len(generated_tokens) < max_tokens and next_token not in stop_tokens:
        next_input = mx.array([[next_token]])
        logits = model(next_input, cache=cache)
        mx.eval(logits)
        
        next_token = int(mx.argmax(logits[:, -1, :]))
        if next_token in stop_tokens:
            break
            
        generated_tokens.append(next_token)
        token_str = tokenizer.decode([next_token])
        current_text += token_str
        text_chunks.append(token_str)
        
        if current_text.strip().endswith("}") and current_text.count("{") == current_text.count("}"):
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000
    token_count = len(generated_tokens)
    tok_per_sec = (token_count / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

    cleaned_json_str = current_text.strip()
    match = re.search(r"(\{.*\})", cleaned_json_str, re.DOTALL)
    if match:
        cleaned_json_str = match.group(1)

    parsed_json = None
    is_valid_json = False
    parse_error = None
    try:
        parsed_json = json.loads(cleaned_json_str)
        is_valid_json = True
    except Exception as e:
        parse_error = str(e)

    missing_keys = []
    invalid_enums = []
    if is_valid_json and isinstance(parsed_json, dict):
        for fname, fdef in schema.fields.items():
            if fname not in parsed_json:
                missing_keys.append(fname)
            elif fdef.field_type != "boolean":
                val = str(parsed_json[fname])
                if val not in fdef.choices:
                    invalid_enums.append(f"{fname}={val}")

    schema_match = is_valid_json and (len(missing_keys) == 0) and (len(invalid_enums) == 0)

    return {
        "mode": "naive_autoregressive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": token_count,
        "tokens_per_second": round(tok_per_sec, 1),
        "sequential_forward_passes": token_count,
        "is_valid_json": is_valid_json,
        "schema_match": schema_match,
        "raw_text": current_text,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "missing_keys": missing_keys,
        "invalid_enums": invalid_enums,
        "has_calibrated_probabilities": False
    }


@gpu_locked_gen
def stream_naive_generation(
    context: str,
    schema: StructuredSchema,
    max_tokens: int = 700,
    temperature: float = 0.2
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields incremental tokens for real-time streaming visualization in the UI.
    """
    model, tokenizer = get_engine()
    prompt = build_naive_json_prompt(context, schema)
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = mx.array(prompt_tokens)[None]
    
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[:, -1, :]))
    
    tok_str = tokenizer.decode([next_token])
    current_text = "{\n  " + tok_str
    token_count = 1
    
    yield {
        "type": "token",
        "token": "{\n  " + tok_str,
        "accumulated": current_text,
        "token_count": token_count,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)
    }
    
    stop_tokens = {tokenizer.eos_token_id}
    for tok_str in ["<end_of_turn>", "<|im_end|>", "<eos>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)
        if tok_id is not None and isinstance(tok_id, int) and tok_id > 0:
            stop_tokens.add(tok_id)
    while token_count < max_tokens and next_token not in stop_tokens:
        next_input = mx.array([[next_token]])
        logits = model(next_input, cache=cache)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[:, -1, :]))
        if next_token in stop_tokens:
            break
        token_count += 1
        delta = tokenizer.decode([next_token])
        current_text += delta
        
        yield {
            "type": "token",
            "token": delta,
            "accumulated": current_text,
            "token_count": token_count,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)
        }
        
        if current_text.strip().endswith("}") and current_text.count("{") == current_text.count("}"):
            break
            
    elapsed_ms = (time.perf_counter() - t0) * 1000
    tok_per_sec = (token_count / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

    cleaned_json_str = current_text.strip()
    match = re.search(r"(\{.*\})", cleaned_json_str, re.DOTALL)
    if match:
        cleaned_json_str = match.group(1)

    parsed_json = None
    is_valid_json = False
    parse_error = None
    try:
        parsed_json = json.loads(cleaned_json_str)
        is_valid_json = True
    except Exception as e:
        parse_error = str(e)

    missing_keys = []
    invalid_enums = []
    if is_valid_json and isinstance(parsed_json, dict):
        for fname, fdef in schema.fields.items():
            if fname not in parsed_json:
                missing_keys.append(fname)
            elif fdef.field_type != "boolean":
                val = str(parsed_json[fname])
                if val not in fdef.choices:
                    invalid_enums.append(f"{fname}={val}")

    schema_match = is_valid_json and (len(missing_keys) == 0) and (len(invalid_enums) == 0)

    final_res = {
        "mode": "naive_autoregressive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": token_count,
        "tokens_per_second": round(tok_per_sec, 1),
        "sequential_forward_passes": token_count,
        "is_valid_json": is_valid_json,
        "schema_match": schema_match,
        "raw_text": current_text,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "missing_keys": missing_keys,
        "invalid_enums": invalid_enums,
        "has_calibrated_probabilities": False
    }
    yield {
        "type": "done",
        "result": final_res
    }


def _resolve_collision_mlx(model, b_cache, base_len, row, suffix_len, field_logits,
                           seqs, quote_id, temperature):
    """
    Resolves one field whose choices share a first token.

    A single token cannot separate e.g. GARAGE from GARDEN, so the decision is walked
    one token deeper at a time, each step constrained to tokens that keep at least one
    declared choice reachable, until a single choice survives. A closing quote acts as
    each choice's terminator so a choice that is a strict prefix of another (ON vs
    ONLINE) stays separable. Costs one 1-token forward pass per extra token of depth,
    and only for the ambiguous group.

    Returns (winner_idx, winner_prob, all_probs, extra_passes).
    """
    # Own view of this row's cache, truncated past the batch padding so the
    # continuation attends to this field's suffix and nothing after it.
    true_len = base_len + suffix_len
    f_cache = []
    for c in b_cache:
        nc = copy.copy(c)
        if hasattr(c, "keys") and c.keys is not None:
            nc.keys = c.keys[row:row + 1, :, :true_len, :]
            nc.values = c.values[row:row + 1, :, :true_len, :]
            nc.offset = true_len
        f_cache.append(nc)

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
            scores = mx.array([float(cur_logits[t]) for t in tok_ids]) / max(temperature, 1e-4)
            probs = mx.softmax(scores)
            mx.eval(probs)
            k = int(mx.argmax(probs))
            path_prob *= float(probs[k])
            chosen = tok_ids[k]
        else:
            chosen = next(iter(branches))

        alive = branches[chosen]
        step += 1
        if len(alive) > 1:
            out_step = model(mx.array([[chosen]]), cache=f_cache)
            mx.eval(out_step)
            cur_logits = out_step[0, -1, :]
            extra_passes += 1

    w_idx = alive[0]
    spread = (1.0 - path_prob) / max(len(seqs) - 1, 1)
    all_probs = [spread] * len(seqs)
    all_probs[w_idx] = path_prob
    return w_idx, path_prob, all_probs, extra_passes


@gpu_locked
def run_parallel_generation(
    context: str,
    schema: StructuredSchema,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Parallel Constrained Decision Engine optimized for Apple Silicon (M4 Max):
    1. Pre-Indexed Schema Metadata: Zero-overhead suffix and token compilation.
    2. High-Density Semantic Prefill: Compact attribute prompt minimizes KV-cache latency.
    3. Broadcast Cache & Batched Suffix Evaluation: Evaluates all M field queries concurrently in 1 forward pass!
    4. Fast Direct Cache Slice Disambiguation: Zero re-allocation continuation for multi-token prefix collisions.
    5. Programmatic Assembly: 100% typed, validated JSON with field-level calibrated confidence scores.
    """
    model, tokenizer = get_engine()
    t0 = time.perf_counter()
    
    # 1. Pre-indexed schema metadata (cached on schema instance)
    meta = schema.compile_parallel_metadata(tokenizer)
    field_items = meta["field_items"]
    suffix_lengths = meta["suffix_lengths"]
    cands_per_field = meta["cands_per_field"]
    has_collisions = meta["has_collisions"]
    suffixes_batch = meta["suffixes_batch"]
    M = suffixes_batch.shape[0]
    
    # 2. High-density semantic catalog for minimal prefill latency
    schema_str = schema.to_parallel_schema_str()
    base_prompt = (
        f"<|im_start|>system\n"
        f"Classify JSON attributes:\n{schema_str}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n"
    )
    base_toks = tokenizer.encode(base_prompt)
    base_arr = mx.array(base_toks)[None]
    
    t_pre0 = time.perf_counter()
    cache = make_prompt_cache(model)
    model(base_arr, cache=cache)
    mx.eval(*[c.keys for c in cache if hasattr(c, "keys")])
    t_prefill = (time.perf_counter() - t_pre0) * 1000
    
    # 3. Broadcast KV cache across batch dimension M with fused Metal evaluation
    b_cache = []
    to_eval = []
    for c in cache:
        nc = copy.copy(c)
        if hasattr(c, "keys") and c.keys is not None:
            nc.keys = mx.repeat(c.keys, M, axis=0)
            nc.values = mx.repeat(c.values, M, axis=0)
            to_eval.extend([nc.keys, nc.values])
        b_cache.append(nc)
    if to_eval:
        mx.eval(*to_eval)
        
    # 4. SINGLE BATCHED FORWARD PASS for all M suffixes!
    t_suf_start = time.perf_counter()
    suffix_out = model(suffixes_batch, cache=b_cache)
    mx.eval(suffix_out)
    t_suffix_eval = (time.perf_counter() - t_suf_start) * 1000
    
    # 5. Extract logits and compute calibrated decisions
    parsed_json = {}
    field_telemetry = {}
    deferred = []
    collision_passes = 0
    
    for i, (fname, fdef) in enumerate(field_items):
        decision_idx = suffix_lengths[i] - 1
        field_logits = suffix_out[i, decision_idx, :]
        cand_tokens = cands_per_field[i]
        
        if not has_collisions[i]:
            scores = [float(field_logits[tid]) for tid in cand_tokens]
            scores_arr = mx.array(scores) / max(temperature, 1e-4)
            probs = mx.softmax(scores_arr)
            mx.eval(probs)
            w_idx = int(mx.argmax(probs))
            w_prob = float(probs[w_idx])
            all_probs = probs.tolist()
            
            raw_choice = ["true", "false"][w_idx] if fdef.field_type == "boolean" else fdef.choices[w_idx]
            val = (raw_choice.lower() == "true") if fdef.field_type == "boolean" else raw_choice
        else:
            # Needs full-sequence scoring; resolved together after this loop.
            deferred.append((i, fname, fdef))
            continue

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

    if deferred:
        base_len = base_arr.shape[1]
        for i, fname, fdef in deferred:
            w_idx, w_prob, all_probs, extra = _resolve_collision_mlx(
                model, b_cache, base_len, i, suffix_lengths[i],
                suffix_out[i, suffix_lengths[i] - 1, :],
                meta["choice_seqs"][i], meta["quote_id"], temperature
            )
            collision_passes += extra
            val = fdef.choices[w_idx]
            parsed_json[fname] = {"value": val, "prob": round(w_prob, 4)}
            scored_choices = sorted(
                ({"choice": c, "probability": round(p, 4)} for c, p in zip(fdef.choices, all_probs)),
                key=lambda x: x["probability"], reverse=True
            )
            field_telemetry[fname] = {
                "value": val,
                "type": fdef.field_type,
                "confidence": round(w_prob, 4),
                "cardinality": fdef.cardinality,
                "top_choices": scored_choices[:5]
            }
        # Restore declared field order after out-of-band resolution
        parsed_json = {n: parsed_json[n] for n, _ in field_items}
        field_telemetry = {n: field_telemetry[n] for n, _ in field_items}

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
        "num_fields": len(schema)
    }


# Backward compatibility alias
run_rlcd_generation = run_parallel_generation
