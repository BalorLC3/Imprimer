"""
Three-panel Gradio interface: Stability Analysis, Optimization, Registry.
"""

import os
import sys
import html
from dotenv import load_dotenv
import json

load_dotenv()

# SSL workaround for certain environments
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not os.path.exists(ssl_cert_file):
    del os.environ["SSL_CERT_FILE"]

engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine"))
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

import gradio as gr
from core.analyzer.stability import analyze as run_stability
from core.optimizer.graph import optimize as run_optimize
from core.registry.prompt_store import best_variant_for_task, init_db
from core.chains.prompt_chain import ModelBackend



TASK_CATEGORIES = [
    "summarize",
    "classify",
    "extract",
    "translate",
    "reasoning",
    "creative_writing",
    "code_generation",
    "rewrite",
    "roleplay",
    "qa",
]

# Change to ModelBackend.OPENAI for OpenAI-backed runs
BACKEND_ID = ModelBackend.OLLAMA

BEST_PROMPT: list[str] = []


CUSTOM_CSS = """
:root {
  --color-background-primary: #ffffff;
  --color-background-secondary: #f8f9fa;
  --color-border-primary: #e5e7eb;
  --color-border-secondary: #e5e7eb;
  --color-border-tertiary: #f3f4f6;
  --color-text-primary: #111827;
  --color-text-secondary: #374151;
  --color-text-tertiary: #6b7280;
  --border-radius-md: 8px;
}

.section-label { font-size: 11px; font-weight: 500; color: var(--color-text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; margin-top: 16px; }

.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 1rem; }
.metric { background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 10px 12px; border: 1px solid var(--color-border-tertiary); }
.metric .label { font-size: 11px; color: var(--color-text-tertiary); margin-bottom: 4px; }
.metric .value { font-size: 20px; font-weight: 500; color: var(--color-text-primary); }
.metric .delta { font-size: 11px; margin-top: 2px; }
.delta.pos { color: #1D9E75; }
.delta.neg { color: #D85A30; }

.timeline { display: flex; flex-direction: column; gap: 4px; margin-bottom: 1rem; }
.iter-row { display: grid; grid-template-columns: 80px 1fr 70px 70px; gap: 8px; align-items: center; padding: 8px 12px; border-radius: var(--border-radius-md); font-size: 12px; border: 1px solid var(--color-border-tertiary); background: var(--color-background-primary); }
.iter-row .iter-label { color: var(--color-text-secondary); font-weight: 500; }
.bar-wrap { background: var(--color-background-secondary); border-radius: 4px; height: 6px; position: relative; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; background: #1D9E75; transition: width 0.4s ease; }
.bar-fill.base { background: #888780; }
.score-val { text-align: right; font-weight: 500; font-size: 12px; }
.score-val.improved { color: #1D9E75; }
.iter-row.running { border-color: #FAC775; background: #FAEEDA22; }
.iter-row.done { border-color: var(--color-border-tertiary); }
.iter-row.pending { opacity: 0.5; }
.spin { display: inline-block; animation: spin 1s linear infinite; font-size: 12px; margin-left: 4px; }
@keyframes spin { to { transform: rotate(360deg); } }

.prompt-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1rem; }
.prompt-box { border: 1px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); padding: 12px; font-size: 13px; background: var(--color-background-primary); }
.prompt-box .box-label { font-size: 10px; font-weight: 600; color: var(--color-text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.prompt-box .text { color: var(--color-text-secondary); line-height: 1.5; white-space: pre-wrap; }
.prompt-box.best .box-label { color: #1D9E75; }
.prompt-box.best { border-color: #5DCAA5; background: #f0fdf455; }

.feedback-card { border-left: 4px solid #5DCAA5; padding: 12px; background: #E1F5EE55; border-radius: 0 var(--border-radius-md) var(--border-radius-md) 0; margin-bottom: 1rem; font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; }
.feedback-card .fb-label { font-size: 10px; font-weight: 600; color: #0F6E56; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }

.run-list { display: flex; flex-direction: column; gap: 8px; }
details.run-item { border: 1px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); overflow: hidden; background: var(--color-background-primary); }
details.run-item > summary { display: flex; align-items: center; gap: 8px; padding: 10px 12px; cursor: pointer; font-size: 13px; font-weight: 500; list-style: none; }
details.run-item > summary::-webkit-details-marker { display: none; }
.run-header .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.run-header .dot.done { background: #1D9E75; }
.run-header .run-score { margin-left: auto; font-weight: 600; color: #1D9E75; }
.run-body { padding: 0px 12px 12px; border-top: 1px solid transparent; font-size: 12px; color: var(--color-text-secondary); line-height: 1.6; white-space: pre-wrap; }
details.run-item[open] > summary { border-bottom: 1px solid var(--color-border-tertiary); margin-bottom: 8px; }

.token-strip { display: flex; flex-wrap: wrap; gap: 4px; padding: 12px; background: var(--color-background-primary); border: 1px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); margin-top: 8px; }
.tok { font-size: 12px; font-family: monospace; padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid transparent; }

.status-bar { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: var(--border-radius-md); font-size: 13px; font-weight: 500; margin-bottom: 1rem; border: 1px solid var(--color-border-tertiary); background: var(--color-background-secondary); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #EF9F27; animation: pulse 1s ease-in-out infinite; flex-shrink: 0; }
.status-dot.done { background: #1D9E75; animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
"""



def _render_token_confidence(token_confidence: list) -> str:
    if not token_confidence:
        return "<p>No token confidence data available.</p>"
    html_str = '<div class="token-strip">'
    for tc in token_confidence:
        certainty = tc.get("certainty", 0.5)
        r = int(216 if certainty < 0.5 else 29)
        g = int(90  if certainty < 0.5 else 158)
        b = int(48  if certainty < 0.5 else 117)
        color = f"#{r:02x}{g:02x}{b:02x}"
        bg    = f"rgba({r},{g},{b},0.15)"
        token = html.escape(tc.get("token", ""))
        logprob = tc.get("logprob", 0)
        html_str += (
            f'<span class="tok" title="certainty={certainty:.3f} logprob={logprob:.3f}" '
            f'style="background:{bg};border-color:{color};">{token}</span>'
        )
    html_str += "</div>"
    return html_str


def build_status_bar(text: str, is_done: bool = False) -> str:
    dot_class = "done" if is_done else ""
    return (
        f'<div class="status-bar">'
        f'<div class="status-dot {dot_class}"></div>'
        f"<span>{html.escape(text)}</span>"
        f"</div>"
    )


def build_metric_html(label: str, value, delta=None) -> str:
    delta_html = ""
    if delta is not None:
        cls  = "pos" if delta > 0 else "neg"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="delta {cls}">{sign}{delta:.3f}</div>'
    val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
    return (
        f'<div class="metric">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{val_str}</div>'
        f"{delta_html}"
        f"</div>"
    )



def run_optimization(
    prompt,
    input_text,
    task,
    model_id,
    expected_output,
    extra_examples_raw,
    n_variants,
    target_score,
    max_iterations,
):
    global BEST_PROMPT

    if not prompt or not task:
        yield (
            "<div class='feedback-card'>Prompt and task are required.</div>",
            "", "", "", "",
        )
        return

    if model_id:
        if BACKEND_ID == ModelBackend.OLLAMA:
            os.environ["OLLAMA_MODEL"] = model_id
        elif BACKEND_ID == ModelBackend.OPENAI:
            os.environ["OPENAI_MODEL"] = model_id

    status_html  = build_status_bar("Initializing optimization graph...", is_done=False)
    metrics_html = (
        f'<div class="metric-row">'
        f'{build_metric_html("Baseline", "---")}'
        f'{build_metric_html("Best so far", "---")}'
        f'{build_metric_html("Target", float(target_score))}'
        f'{build_metric_html("Cycles", f"0 / {max_iterations}")}'
        f"</div>"
    )
    prompt_html = (
        f'<div class="prompt-compare">'
        f'<div class="prompt-box"><div class="box-label">Original</div>'
        f'<div class="text">{html.escape(prompt)}</div></div>'
        f'<div class="prompt-box best"><div class="box-label">Optimized</div>'
        f"<div class='text'>Waiting for first cycle...</div></div>"
        f"</div>"
    )
    timeline_html = (
        "<div class='timeline'>"
        "<div class='iter-row running'>"
        "<span class='iter-label'>Cycle 1 <span class='spin'>&#x21BB;</span></span>"
        "<div class='bar-wrap'><div class='bar-fill base' style='width:0%'></div></div>"
        "<span class='score-val'>-</span><span class='score-val'>-</span>"
        "</div></div>"
    )
    feedback_html = "<div class='feedback-card'>Waiting for first cycle reflection...</div>"

    yield status_html, metrics_html, timeline_html, prompt_html, feedback_html

    # Per-cycle score history: list of (cycle_reachability, best_reachability, improvement)
    cycle_history: list[tuple[float, float, float]] = []

    final_result = None
    try: 
        if extra_examples_raw.strip():
            extra_samples = json.loads(extra_examples_raw)
    except Exception:
        pass

    try:
        optimizer_output = run_optimize(
            task=task,
            base_prompt=prompt,
            input_example=input_text,
            expected_output=expected_output,
            extra_samples=extra_examples_raw,
            n_variants=int(n_variants),
            backend=BACKEND_ID,
            target_score=float(target_score),
            max_iterations=int(max_iterations),
        )

        for step_result in optimizer_output:
            final_result = step_result
            iteration    = step_result.get("iterations_completed", step_result.get("current_iteration", 1))
            base_s       = step_result.get("baseline_score", 0.0)
            best_s       = step_result.get("best_score", 0.0)
            cycle_s      = step_result.get("cycle_reachability", best_s)   # this cycle's actual score
            improv       = step_result.get("improvement", 0.0)
            curr_p       = step_result.get("best_prompt", "")
            fb_str       = step_result.get("feedback", "")
            g_mean       = step_result.get("grpo_group_mean", 0.0)
            BEST_PROMPT.append(curr_p)

            # Record this cycle: (cycle score, global best at this point, delta)
            cycle_history.append((cycle_s, best_s, round(cycle_s - base_s, 4)))

            status_html  = build_status_bar(
                f"Cycle {iteration} of {max_iterations} complete",
                is_done=False,
            )
            metrics_html = (
                f'<div class="metric-row">'
                f'{build_metric_html("Baseline", base_s)}'
                f'{build_metric_html("This cycle", cycle_s, delta=round(cycle_s - base_s, 4))}'
                f'{build_metric_html("Best overall", best_s, delta=improv)}'
                f'{build_metric_html("Target", float(target_score))}'
                f'{build_metric_html("Group mean", g_mean)}'
                f"</div>"
            )

            # Timeline: completed cycles show their OWN score; running cycle shows spinner
            tl = '<div class="timeline">'
            for i in range(1, int(max_iterations) + 1):
                if i <= len(cycle_history):
                    c_reach, b_reach, c_delta = cycle_history[i - 1]
                    pct      = min(100, int(c_reach * 100))
                    is_best  = abs(c_reach - b_reach) < 1e-4  # this cycle WAS the best
                    delta_cls = "improved" if c_delta >= 0 else ""
                    delta_str = f"+{c_delta:.3f}" if c_delta >= 0 else f"{c_delta:.3f}"
                    tl += (
                        f'<div class="iter-row done">'
                        f'<span class="iter-label">Cycle {i}{"  ★" if is_best else ""}</span>'
                        f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div>'
                        f'<span class="score-val {delta_cls}">{c_reach:.3f}</span>'
                        f'<span class="score-val {delta_cls}">{delta_str}</span>'
                        f"</div>"
                    )
                elif i == len(cycle_history) + 1:
                    tl += (
                        f'<div class="iter-row running">'
                        f'<span class="iter-label">Cycle {i} <span class="spin">&#x21BB;</span></span>'
                        f'<div class="bar-wrap"><div class="bar-fill base" style="width:0%"></div></div>'
                        f'<span class="score-val">...</span>'
                        f'<span class="score-val">...</span>'
                        f"</div>"
                    )
                else:
                    tl += (
                        f'<div class="iter-row pending">'
                        f'<span class="iter-label">Cycle {i}</span>'
                        f'<div class="bar-wrap"><div class="bar-fill" style="width:0%"></div></div>'
                        f'<span class="score-val">-</span><span class="score-val">-</span>'
                        f"</div>"
                    )
            tl += "</div>"

            prompt_html = (
                f'<div class="prompt-compare">'
                f'<div class="prompt-box"><div class="box-label">Original</div>'
                f'<div class="text">{html.escape(prompt)}</div></div>'
                f'<div class="prompt-box best">'
                f'<div class="box-label">Best so far (Cycle {iteration})</div>'
                f'<div class="text">{html.escape(curr_p)}</div></div>'
                f"</div>"
            )
            feedback_html = (
                f'<div class="feedback-card">'
                f'<div class="fb-label">Cycle {iteration} Reflection</div>'
                f"{html.escape(fb_str) if fb_str else 'Generating new variations and scoring...'}"
                f"</div>"
            )

            yield status_html, metrics_html, tl, prompt_html, feedback_html

    except Exception as e:
        err = (
            "<div class='feedback-card' style='border-color:red;'>"
            "<div class='fb-label' style='color:red;'>Error</div>"
            f"{html.escape(str(e))}</div>"
        )
        yield build_status_bar("Optimization failed", True), "", "", "", err
        return

    result         = final_result or {}
    final_iter     = result.get("iterations_completed", result.get("current_iteration", max_iterations))
    base_s         = result.get("baseline_score", 0.0)
    best_s         = result.get("best_score", 0.0)
    improv         = result.get("improvement", 0.0)
    g_mean         = result.get("grpo_group_mean", 0.0)
    status_msg     = (
        "Target score reached - optimization complete"
        if result.get("target_reached")
        else "Iteration cap reached - optimization finished"
    )

    status_html  = build_status_bar(status_msg, is_done=True)
    metrics_html = (
        f'<div class="metric-row">'
        f'{build_metric_html("Baseline", base_s)}'
        f'{build_metric_html("Final best", best_s, delta=improv)}'
        f'{build_metric_html("Target", float(target_score))}'
        f'{build_metric_html("Group mean", g_mean)}'
        f'{build_metric_html("Cycles", f"{final_iter} / {max_iterations}")}'
        f"</div>"
    )

    # Final timeline — all cycles done, no spinner
    tl = '<div class="timeline">'
    for i in range(1, len(cycle_history) + 1):
        c_reach, b_reach, c_delta = cycle_history[i - 1]
        pct       = min(100, int(c_reach * 100))
        is_best   = abs(c_reach - b_reach) < 1e-4
        delta_cls = "improved" if c_delta >= 0 else ""
        delta_str = f"+{c_delta:.3f}" if c_delta >= 0 else f"{c_delta:.3f}"
        tl += (
            f'<div class="iter-row done">'
            f'<span class="iter-label">Cycle {i}{"  ★" if is_best else ""}</span>'
            f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div>'
            f'<span class="score-val {delta_cls}">{c_reach:.3f}</span>'
            f'<span class="score-val {delta_cls}">{delta_str}</span>'
            f"</div>"
        )
    tl += "</div>"

    best_p = result.get("best_prompt", "")
    prompt_html = (
        f'<div class="prompt-compare">'
        f'<div class="prompt-box"><div class="box-label">Original</div>'
        f'<div class="text">{html.escape(prompt)}</div></div>'
        f'<div class="prompt-box best"><div class="box-label">Final Optimized Prompt</div>'
        f'<div class="text">{html.escape(best_p)}</div></div>'
        f"</div>"
    )
    fb = result.get("feedback", "")
    feedback_html = (
        f'<div class="feedback-card">'
        f'<div class="fb-label">Final Reflection</div>'
        f"{html.escape(fb) if fb else 'Target reached or maximum iterations exhausted.'}"
        f"</div>"
    )

    yield status_html, metrics_html, tl, prompt_html, feedback_html



def run_analysis(prompt, input_text, task, model_id, n_runs, temperature):
    if not prompt or not task:
        return (
            "<div class='feedback-card'>Prompt and task are required.</div>",
            "", "", "", None,
        )

    if model_id:
        if BACKEND_ID == ModelBackend.OLLAMA:
            os.environ["OLLAMA_MODEL"] = model_id
        elif BACKEND_ID == ModelBackend.OPENAI:
            os.environ["OPENAI_MODEL"] = model_id

    try:
        result = run_stability(
            prompt=prompt,
            input_text=input_text,
            task=task,
            backend=BACKEND_ID,
            n_runs=int(n_runs),
            temperature=float(temperature),
        )
    except Exception as e:
        err = (
            "<div class='feedback-card' style='border-color:red;'>"
            "<div class='fb-label' style='color:red;'>Error</div>"
            f"{html.escape(str(e))}</div>"
        )
        return build_status_bar("Analysis failed", True), "", err, "", None

    score      = result.stability_score
    status_html = build_status_bar(
        f"Analysis complete - {n_runs} runs, stability score {score:.3f}", is_done=True
    )
    metrics_html = (
        f'<div class="metric-row">'
        f'{build_metric_html("Stability Score", score)}'
        f'{build_metric_html("Avg Reachability", result.avg_reachability)}'
        f'{build_metric_html("Avg Similarity", result.avg_similarity)}'
        f'{build_metric_html("Variance", result.variance)}'
        f"</div>"
        f'<div class="feedback-card" style="margin-top:1rem;">'
        f'<div class="fb-label">Recommendation</div>'
        f"{html.escape(getattr(result, 'recommendation', ''))}"
        f"</div>"
    )

    outputs_html = '<div class="run-list">'
    for i, out in enumerate(result.outputs):
        open_attr = "open" if i == 0 else ""
        outputs_html += (
            f'<details class="run-item" {open_attr}>'
            f'<summary class="run-header">'
            f'<div class="dot done"></div>'
            f"<span>Run {i + 1}</span>"
            f"</summary>"
            f'<div class="run-body">{html.escape(out)}</div>'
            f"</details>"
        )
    outputs_html += "</div>"

    token_html = _render_token_confidence([
        {"token": tc.token, "certainty": tc.certainty, "logprob": tc.logprob}
        for tc in result.token_confidence
    ])

    return status_html, metrics_html, outputs_html, token_html, result



def query_best(task, limit):
    if not task:
        return "Task is required."
    try:
        result = best_variant_for_task(task, limit=int(limit))
        if not result.get("task"):
            return f"No evaluations found for task '{task}'."
        return (
            f"**Task:** {result['task']}\n"
            f"**Evaluations sampled:** {result['evaluations_sampled']}\n"
            f"**Avg score:** {result['avg_score']:.4f}\n\n"
            f"**Best prompt:**\n{result['best_template']}"
        )
    except Exception as e:
        return str(e)


OLLAMA_MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:7b", "llama3.2:7b", "llama3.2:latest"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

with gr.Blocks(title="Imprimer - LLM Prompt Control") as demo:
    gr.Markdown("""
# Imprimer - LLM Prompt Control Platform

> *Prompts don't instruct a unified mind - they activate configurations within it.*
> Imprimer makes those activations **measurable**, **comparable**, and **improvable**.

Optimizer: **GRPO + RiOT residuals** (group-relative reward shaping + semantic drift prevention)
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            prompt_input = gr.Textbox(
                label="Prompt template",
                placeholder="Summarize this in one sentence: {input}",
                lines=3,
            )
            input_text = gr.Textbox(
                label="Input text (optional)",
                placeholder="The text your prompt will process...",
                lines=3,
            )
            task_input = gr.Dropdown(
                label="Task type",
                choices=TASK_CATEGORIES,
                value="summarize",
                allow_custom_value=True,
            )
            if BACKEND_ID == ModelBackend.OPENAI:
                model_id = gr.Dropdown(
                    label="OpenAI Model",
                    choices=OPENAI_MODELS,
                    value=OPENAI_MODELS[0],
                    allow_custom_value=True,
                )
            else:
                model_id = gr.Dropdown(
                    label="Ollama Model",
                    choices=OLLAMA_MODELS,
                    value="llama3.2:latest",
                    allow_custom_value=True,
                )

    gr.Markdown("---")

    with gr.Tabs():

        with gr.TabItem("Stability Analysis"):
            with gr.Row():
                n_runs = gr.Slider(
                    minimum=2, maximum=5, value=3, step=1,
                    label="Number of runs (N samples)",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature",
                )
            analyze_btn = gr.Button("Analyze Prompt", variant="primary")

            stab_status_out  = gr.HTML()
            stab_metrics_out = gr.HTML()
            gr.HTML('<div class="section-label">Outputs - Expand to read</div>')
            stab_outputs_out = gr.HTML()
            gr.HTML('<div class="section-label">Token confidence - First output</div>')
            stab_token_out   = gr.HTML()
            _analysis_state  = gr.State()

            analyze_btn.click(
                fn=run_analysis,
                inputs=[prompt_input, input_text, task_input, model_id, n_runs, temperature],
                outputs=[stab_status_out, stab_metrics_out, stab_outputs_out,
                         stab_token_out, _analysis_state],
            )


        with gr.TabItem("Optimization"):
            gr.Markdown("""
Run **GRPO + RiOT** optimization inside a LangGraph control loop.
Each cycle generates a group of prompt variants, applies group-relative reward shaping,
and uses RiOT residual connections to carry proven constraints forward across cycles.
""")
            with gr.Row():
                expected_output = gr.Textbox(
                    label="Reference output for similarity scoring",
                    placeholder="e.g. 'Positive' (best for classify/extract). Leave blank for creative tasks.",
                    lines=2,
                )
                extra_examples_input = gr.Textbox(
                    label='Extra examples (optional JSON: [{"input":"...","expected":"..."}])',
                    placeholder='[{"input": "Apple stock dropped", "expected": "bearish"}]',
                    lines=2,
                )
            with gr.Row():
                n_variants = gr.Slider(
                    minimum=2, maximum=8, value=4, step=1,
                    label="Variants per cycle (GRPO group size)",
                )
                target_score = gr.Slider(
                    minimum=0.5, maximum=0.97, value=0.80, step=0.01,
                    label="Target reachability score",
                )
                max_iter = gr.Slider(
                    minimum=2, maximum=10, value=3, step=1,
                    label="Max graph iterations",
                )

            optimize_btn = gr.Button("Optimize Prompt", variant="primary")

            opt_status_out   = gr.HTML()
            gr.HTML('<div class="section-label">Score progress</div>')
            opt_metrics_out  = gr.HTML()
            gr.HTML('<div class="section-label">Iteration timeline</div>')
            opt_timeline_out = gr.HTML()
            gr.HTML('<div class="section-label">Prompt comparison</div>')
            opt_prompt_out   = gr.HTML()
            gr.HTML('<div class="section-label">AI reflection</div>')
            opt_feedback_out = gr.HTML()

            optimize_btn.click(
                fn=run_optimization,
                inputs=[
                    prompt_input, input_text, task_input, model_id,
                    expected_output, extra_examples_input, n_variants, target_score, max_iter,
                ],
                outputs=[
                    opt_status_out, opt_metrics_out, opt_timeline_out,
                    opt_prompt_out, opt_feedback_out,
                ],
            )

        with gr.TabItem("Registry"):
            gr.Markdown("""
Query the registry for the best known prompt for a given task,
based on the **average historical score** across all evaluations.
""")
            with gr.Row():
                registry_task = gr.Dropdown(
                    label="Task to search",
                    choices=TASK_CATEGORIES,
                    value="summarize",
                    allow_custom_value=True,
                )
                registry_limit = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Evaluations to sample",
                )
            registry_btn = gr.Button("Query Registry", variant="secondary")
            registry_out = gr.Markdown()

            registry_btn.click(
                fn=query_best,
                inputs=[registry_task, registry_limit],
                outputs=[registry_out],
            )

    gr.Markdown("---\n**Imprimer** · Karim Luna")


if __name__ == "__main__":
    init_db()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )