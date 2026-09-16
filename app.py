"""
Hugging Face Spaces Interactive Demo for Parallel Constrained Decoding.
Optimized for Nvidia ZeroGPU (A10G) and PyTorch.
"""

import os
import json
import time
from typing import Dict, Any, Generator

import gradio as gr
from core.schema import StructuredSchema
from core.engine import run_parallel_generation, run_naive_generation

# ZeroGPU decorator support
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=60)
except Exception:
    def gpu_decorator(fn):
        return fn

# Load presets from presets/ directory
PRESETS = {}
presets_dir = os.path.join(os.path.dirname(__file__), "presets")
if os.path.exists(presets_dir):
    for fname in sorted(os.listdir(presets_dir)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(presets_dir, fname), "r") as f:
                    data = json.load(f)
                    title = data.get("title", fname)
                    PRESETS[title] = {
                        "context": data.get("context", ""),
                        "schema": json.dumps(data.get("schema", {}), indent=2)
                    }
            except Exception as e:
                print(f"Error loading {fname}: {e}")

preset_titles = list(PRESETS.keys())
default_title = preset_titles[0] if preset_titles else None
default_context = PRESETS[default_title]["context"] if default_title else ""
default_schema = PRESETS[default_title]["schema"] if default_title else "{}"


@gpu_decorator
def run_comparison(context_str: str, schema_json_str: str):
    if not context_str or not context_str.strip():
        yield (
            "<div style='color: #dc2626; font-weight: 600; padding: 6px 12px;'>Please provide a context prompt.</div>",
            "{}",
            "0.0 ms",
            "{}",
            "0.0 ms"
        )
        return

    try:
        schema_dict = json.loads(schema_json_str)
        schema = StructuredSchema(schema_dict)
    except Exception as e:
        yield (
            f"<div style='color: #dc2626; font-weight: 600; padding: 6px 12px;'>Invalid Schema JSON: {e}</div>",
            "{}",
            "0.0 ms",
            "{}",
            "0.0 ms"
        )
        return

    try:
        # 1. Run Parallel Constrained Decoding first
        parallel_res = run_parallel_generation(context_str, schema)
        parallel_ms = parallel_res["elapsed_ms"]
        parallel_json_str = json.dumps(parallel_res["parsed_json"], indent=2)
        parallel_time_badge = f"{parallel_ms:.1f} ms"

        summary_intermediate = f"""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 9999px; padding: 6px 16px; display: inline-flex; align-items: center; gap: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px;">
            <span style="color: #16a34a; font-weight: 700;">Parallel Done: {parallel_time_badge}</span>
            <span style="color: #94a3b8;">·</span>
            <span style="color: #64748b;">Evaluating normal autoregressive baseline...</span>
        </div>
        """
        
        yield (
            summary_intermediate,
            parallel_json_str,
            parallel_time_badge,
            "// Running sequential autoregressive baseline forward passes...",
            "Evaluating..."
        )

        # 2. Run Naive generation baseline
        naive_res = run_naive_generation(context_str, schema)
        naive_ms = naive_res["elapsed_ms"]
        naive_json_str = json.dumps(naive_res["parsed_json"], indent=2) if naive_res.get("parsed_json") else naive_res.get("raw_text", "")
        naive_time_badge = f"{naive_ms:.1f} ms"

        speedup = round(naive_ms / max(parallel_ms, 1.0), 1)

        final_summary_html = f"""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 9999px; padding: 8px 20px; display: inline-flex; align-items: center; gap: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <span style="color: #16a34a; font-weight: 800; font-size: 16px; letter-spacing: 0.5px;">{speedup}x FASTER</span>
            <span style="color: #cbd5e1; font-weight: 600;">·</span>
            <span style="color: #334155; font-weight: 600; font-family: monospace;">{parallel_time_badge} vs {naive_time_badge}</span>
        </div>
        """

        yield (
            final_summary_html,
            parallel_json_str,
            parallel_time_badge,
            naive_json_str,
            naive_time_badge
        )
    except Exception as err:
        import traceback
        err_msg = f"{err}\n{traceback.format_exc()}"
        yield (
            f"<div style='color: #dc2626; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 14px; font-family: monospace; font-size: 13px;'>Error: {err}</div>",
            "{}",
            "0.0 ms",
            f"Error details:\n{err_msg}",
            "0.0 ms"
        )


with gr.Blocks(title="Parallel Constrained Decision Engine") as demo:
    gr.Markdown("# Parallel Constrained vs Normal Inference (Qwen2.5 1.5B)")
    gr.Markdown("Parallel Constrained Decoding evaluates all schema fields simultaneously against broadcast prefix KV-cache states, delivering substantial latency reductions with 100% schema adherence.")

    with gr.Row():
        preset_dropdown = gr.Dropdown(
            choices=preset_titles,
            value=default_title,
            label="Select Preset Scenario",
            scale=4
        )
        btn_run = gr.Button("⚡ Run Comparison", variant="primary", scale=1)

    summary_banner = gr.HTML(value="")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Parallel Constrained (Qwen2.5 1.5B)")
            timer_parallel = gr.Textbox(label="Elapsed Time", value="0.0 ms", interactive=False, max_lines=1)
            output_parallel = gr.Code(label="Parallel JSON (Values + Calibrated Probabilities)", language="json", interactive=False, lines=18)
        
        with gr.Column(scale=1):
            gr.Markdown("### Normal Inference (Qwen2.5 1.5B)")
            timer_naive = gr.Textbox(label="Elapsed Time", value="0.0 ms", interactive=False, max_lines=1)
            output_naive = gr.Code(label="Autoregressive JSON Output", language="json", interactive=False, lines=18)

    with gr.Accordion("Inspect Context Document & Schema Definition", open=False):
        context_input = gr.Textbox(
            label="Context Document",
            value=default_context,
            lines=6
        )
        schema_input = gr.Code(
            label="Schema Definition (JSON)",
            value=default_schema,
            language="json",
            lines=10
        )

    def on_preset_change(title):
        if title in PRESETS:
            return PRESETS[title]["context"], PRESETS[title]["schema"]
        return "", "{}"

    preset_dropdown.change(
        fn=on_preset_change,
        inputs=[preset_dropdown],
        outputs=[context_input, schema_input]
    )

    btn_run.click(
        fn=run_comparison,
        inputs=[context_input, schema_input],
        outputs=[summary_banner, output_parallel, timer_parallel, output_naive, timer_naive]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
