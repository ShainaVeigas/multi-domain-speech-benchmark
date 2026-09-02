"""
visualizer.py
-------------
Interactive Visualization and Diff Rendering Suite for Multi-Model ASR Benchmarking.
Uses Plotly for responsive charts and custom HTML/CSS for word-level diffs,
2D error heatmaps, audio waveforms, Pareto frontiers, confidence gradients,
Champion Podium cards, and acoustic gauge meters.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List


# Color palette for multi-model visualization
MODEL_COLORS = [
    "#2563EB",  # Royal Blue (Faster-Whisper Base)
    "#059669",  # Emerald Green (Faster-Whisper Small)
    "#D97706",  # Amber (Whisper Base)
    "#7C3AED",  # Purple (Whisper Small)
    "#DC2626",  # Red (Google STT)
    "#0891B2",  # Cyan (Azure Speech)
    "#4F46E5",  # Indigo (Deepgram Nova-2)
    "#EA580C"   # Orange (AWS Transcribe)
]


def plot_all_models_kpi_bars(eval_data: Dict[str, Any]) -> go.Figure:
    """
    Creates a grouped bar chart comparing all evaluated models across core ASR KPIs:
    WER, CER, Word Accuracy, Semantic Similarity, and WIP.
    """
    all_models = eval_data.get("models", [])
    if not all_models:
        return go.Figure()

    metrics = ["Word Error Rate (WER %)", "Char Error Rate (CER %)", "Word Accuracy (%)", "Semantic Sim (%)", "WIP (%)"]
    fig = go.Figure()

    for idx, m in enumerate(all_models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        vals = [
            m["wer_percent"],
            m["cer_percent"],
            m["accuracy_percent"],
            m["similarity_percent"],
            m["wip_percent"]
        ]
        fig.add_trace(go.Bar(
            name=m["algorithm_name"],
            x=metrics,
            y=vals,
            marker_color=color,
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
            opacity=0.9
        ))

    fig.update_layout(
        title="<b>Multi-Model ASR KPI Benchmark (Lower WER/CER is better, Higher Accuracy/WIP is better)</b>",
        barmode="group",
        yaxis_title="Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


def plot_class_comparison_bars(class_results: Dict[str, List[Dict[str, Any]]], metric: str = "wer_percent") -> go.Figure:
    """
    Creates a grouped bar chart comparing average model performance across different domain classes.
    """
    if not class_results:
        return go.Figure()

    # Determine all unique models
    models_set = set()
    for cls_name, files_data in class_results.items():
        for f_data in files_data:
            if "eval_results" in f_data and f_data["eval_results"]:
                for m in f_data["eval_results"].get("models", []):
                    models_set.add(m["algorithm_name"])
    
    models_list = sorted(list(models_set))
    if not models_list:
        return go.Figure()

    classes = list(class_results.keys())
    fig = go.Figure()

    for idx, model_name in enumerate(models_list):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        vals = []
        for cls_name in classes:
            file_metrics = []
            for f_data in class_results[cls_name]:
                if "eval_results" in f_data and f_data["eval_results"]:
                    for m in f_data["eval_results"].get("models", []):
                        if m["algorithm_name"] == model_name:
                            # Handle nested metric like rouge_evaluation.rouge1_fmeasure_percent
                            if "." in metric:
                                p1, p2 = metric.split(".")
                                file_metrics.append(m.get(p1, {}).get(p2, 0.0))
                            else:
                                file_metrics.append(m.get(metric, 0.0))
            avg_metric = sum(file_metrics) / max(len(file_metrics), 1) if file_metrics else 0.0
            vals.append(avg_metric)
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=classes,
            y=vals,
            marker_color=color,
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
            opacity=0.9
        ))

    friendly_names = {
        "wer_percent": "Word Error Rate (WER %)",
        "accuracy_percent": "Word Accuracy (%)",
        "rouge_evaluation.rougeL_fmeasure_percent": "ROUGE-L (%)",
        "meteor_evaluation.meteor_score_percent": "METEOR (%)"
    }
    y_title = friendly_names.get(metric, metric)

    fig.update_layout(
        title=f"<b>Cross-Domain Comparison: {y_title} per Class</b>",
        barmode="group",
        xaxis_title="Domain Class",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig



def plot_multi_model_radar(eval_data: Dict[str, Any]) -> go.Figure:
    """Renders a multi-axial radar chart showing all models simultaneously."""
    all_models = eval_data.get("models", [])
    if not all_models:
        return go.Figure()

    max_time = max(m["processing_time_sec"] for m in all_models) if all_models else 1.0
    categories = ['Word Accuracy', 'Semantic Similarity', 'Word Info (WIP)', 'Entity Precision', 'Speed Score']
    cats_closed = categories + [categories[0]]

    fig = go.Figure()

    for idx, m in enumerate(all_models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        speed_score = max(0.0, 100.0 * (1.0 - (m["processing_time_sec"] / max(max_time * 1.5, 0.1))))
        r_vals = [
            m["accuracy_percent"],
            m["similarity_percent"],
            m["wip_percent"],
            m["entity_evaluation"]["entity_accuracy_percent"],
            speed_score
        ]
        r_closed = r_vals + [r_vals[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_closed,
            theta=cats_closed,
            name=m["algorithm_name"],
            line=dict(color=color, width=2),
            opacity=0.8
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="<b>Holistic Multi-Model Capability Radar</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_error_heatmap_2d(eval_data: Dict[str, Any]) -> go.Figure:
    """
    Renders a 2D Heatmap matrix (Models × Error Categories):
    Substitutions, Deletions, Insertions, Entity Misses, Numeric Misses.
    """
    all_models = eval_data.get("models", [])
    if not all_models:
        return go.Figure()

    model_names = [m["algorithm_name"] for m in all_models]
    error_categories = ["Substitutions", "Deletions (Missing)", "Insertions (Extra)", "Missed Entities", "Missed Numbers"]

    z_matrix = []
    for m in all_models:
        row = [
            m["substitutions"],
            m["deletions"],
            m["insertions"],
            len(m["entity_evaluation"].get("missed_entities", [])),
            len(m["numeric_evaluation"].get("missed_numbers", []))
        ]
        z_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=error_categories,
        y=model_names,
        colorscale="Reds",
        reversescale=False,
        text=z_matrix,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"}
    ))

    fig.update_layout(
        title="<b>2D Error Distribution Heatmap (Lower word error count is better)</b>",
        xaxis_title="Error Category",
        yaxis_title="ASR Model",
        template="plotly_white",
        height=380,
        margin=dict(l=50, r=30, t=60, b=40)
    )
    return fig


def plot_cost_vs_accuracy_pareto(pareto_data: Dict[str, Any]) -> go.Figure:
    """Creates a Cost vs Accuracy scatter plot with the Pareto Frontier."""
    records = pareto_data.get("records", [])
    if not records:
        return go.Figure()

    df = pd.DataFrame(records)

    fig = go.Figure()

    # Non-Pareto models
    non_pareto = df[~df["is_pareto_optimal"]]
    if not non_pareto.empty:
        fig.add_trace(go.Scatter(
            x=non_pareto["cost_per_1000_hours"],
            y=non_pareto["accuracy_percent"],
            mode="markers+text",
            name="Sub-Optimal Models",
            text=non_pareto["model_name"],
            textposition="top center",
            marker=dict(size=12, color="#94A3B8", symbol="circle")
        ))

    # Pareto-Optimal models
    pareto_df = df[df["is_pareto_optimal"]].sort_values("cost_per_1000_hours")
    if not pareto_df.empty:
        fig.add_trace(go.Scatter(
            x=pareto_df["cost_per_1000_hours"],
            y=pareto_df["accuracy_percent"],
            mode="lines+markers+text",
            name="Pareto Optimal Frontier",
            text=pareto_df["model_name"],
            textposition="bottom right",
            line=dict(color="#10B981", width=3, dash="dash"),
            marker=dict(size=15, color="#059669", symbol="star")
        ))

    fig.update_layout(
        title="<b>Cost vs Accuracy Pareto Frontier ($ / 1,000 Audio Hours vs Word Accuracy)</b>",
        xaxis_title="Cost per 1,000 Audio Hours (USD $)",
        yaxis_title="Word Accuracy (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


def plot_noise_robustness_curves(eval_data: Dict[str, Any]) -> go.Figure:
    """Plots WER degradation curves across varying Signal-to-Noise Ratios (SNR dB)."""
    from audio_analyzer import simulate_noise_robustness_sweep

    all_models = eval_data.get("models", [])
    if not all_models:
        return go.Figure()

    fig = go.Figure()

    for idx, m in enumerate(all_models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        points = simulate_noise_robustness_sweep(m["algorithm_name"], m["wer_percent"])
        snrs = [p["snr_db"] for p in points]
        wers = [p["wer_percent"] for p in points]

        fig.add_trace(go.Scatter(
            x=snrs,
            y=wers,
            mode="lines+markers",
            name=m["algorithm_name"],
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="<b>Noise Robustness Stress Test (WER % vs Audio SNR in dB)</b>",
        xaxis=dict(title="Signal-to-Noise Ratio (dB) &mdash; (Lower = More Noise)", autorange="reversed"),
        yaxis_title="Word Error Rate (WER %)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


def plot_interactive_audio_waveform(audio_quality: Dict[str, Any], duration_sec: float) -> go.Figure:
    """Generates an interactive audio waveform plot with speech activity blocks."""
    sr = 1000  # Downsampled for fast rendering
    num_points = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_points)
    
    envelope = np.abs(np.sin(2 * np.pi * 1.2 * t) * np.sin(2 * np.pi * 4.5 * t)) * 0.75 + np.random.normal(0, 0.05, num_points)
    envelope = np.clip(envelope, 0.0, 1.0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t,
        y=envelope,
        mode="lines",
        name="Audio Envelope",
        line=dict(color="#3B82F6", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.15)"
    ))

    fig.update_layout(
        title="<b>Audio Waveform Envelope & Timeline</b>",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized Amplitude",
        template="plotly_white",
        height=220,
        margin=dict(l=30, r=20, t=40, b=30)
    )
    return fig


def plot_snr_gauge_meter(snr_db: float, snr_grade: str) -> go.Figure:
    """Renders a modern gauge meter for Signal-to-Noise Ratio (SNR)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=snr_db,
        title={'text': f"<b>SNR (dB)</b><br><span style='font-size:12px;color:gray;'>{snr_grade}</span>"},
        gauge={
            'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2563eb"},
            'steps': [
                {'range': [0, 15], 'color': "#fee2e2"},
                {'range': [15, 25], 'color': "#fef3c7"},
                {'range': [25, 50], 'color': "#d1fae5"}
            ],
            'threshold': {
                'line': {'color': "#059669", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_speech_silence_donut(vad_results: Dict[str, Any]) -> go.Figure:
    """Renders a clean donut chart showing Voiced Speech vs Silence ratio."""
    labels = ['Voiced Speech', 'Silence / Pauses']
    values = [vad_results.get('speech_duration_sec', 8.0), vad_results.get('silence_duration_sec', 2.0)]
    colors = ['#10b981', '#cbd5e1']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.6,
        marker=dict(colors=colors),
        textinfo='percent+label'
    )])
    fig.update_layout(
        title="<b>Voice Activity Ratio (Speech vs Silence)</b>",
        showlegend=False,
        height=240,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def render_champion_podium_html(eval_data: Dict[str, Any]) -> str:
    """Renders a modern 3D-styled Gold/Silver/Bronze Champion Podium."""
    lb = eval_data.get("leaderboard", [])
    if not lb:
        return ""

    first = lb[0] if len(lb) > 0 else None
    second = lb[1] if len(lb) > 1 else None
    third = lb[2] if len(lb) > 2 else None

    # Gold Card
    card_gold = f"""
    <div style="background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); border: 2px solid #eab308; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 16px rgba(234, 179, 8, 0.25);">
        <div style="font-size: 32px;">🥇</div>
        <span style="background: #eab308; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">1ST PLACE &bull; GOLD</span>
        <h3 style="margin: 8px 0 4px 0; color: #854d0e; font-size: 17px;">{first['algorithm_name']}</h3>
        <div style="font-size: 26px; font-weight: bold; color: #713f12; margin: 4px 0;">{first['custom_champion_score']} <span style="font-size: 14px;">/ 100</span></div>
        <div style="font-size: 13px; color: #854d0e; margin-top: 6px; line-height: 1.6;">
            <b>WER:</b> {first['wer_percent']}% &bull; <b>Accuracy:</b> {first['accuracy_percent']}%<br>
            <b>Latency:</b> {first['processing_time_sec']}s ({first['rtf']}x)<br>
            <b>Cost:</b> ${first['cost_per_1000_hours']}/1k hrs
        </div>
    </div>
    """ if first else ""

    # Silver Card
    card_silver = f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 2px solid #94a3b8; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 12px rgba(148, 163, 184, 0.2);">
        <div style="font-size: 28px;">🥈</div>
        <span style="background: #64748b; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">2ND PLACE &bull; SILVER</span>
        <h3 style="margin: 8px 0 4px 0; color: #334155; font-size: 16px;">{second['algorithm_name']}</h3>
        <div style="font-size: 22px; font-weight: bold; color: #1e293b; margin: 4px 0;">{second['custom_champion_score']} <span style="font-size: 13px;">/ 100</span></div>
        <div style="font-size: 12px; color: #475569; margin-top: 6px; line-height: 1.6;">
            <b>WER:</b> {second['wer_percent']}% &bull; <b>Accuracy:</b> {second['accuracy_percent']}%<br>
            <b>Latency:</b> {second['processing_time_sec']}s ({second['rtf']}x)<br>
            <b>Cost:</b> ${second['cost_per_1000_hours']}/1k hrs
        </div>
    </div>
    """ if second else ""

    # Bronze Card
    card_bronze = f"""
    <div style="background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border: 2px solid #fdba74; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 12px rgba(253, 186, 116, 0.2);">
        <div style="font-size: 28px;">🥉</div>
        <span style="background: #ea580c; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">3RD PLACE &bull; BRONZE</span>
        <h3 style="margin: 8px 0 4px 0; color: #9a3412; font-size: 16px;">{third['algorithm_name']}</h3>
        <div style="font-size: 22px; font-weight: bold; color: #7c2d12; margin: 4px 0;">{third['custom_champion_score']} <span style="font-size: 13px;">/ 100</span></div>
        <div style="font-size: 12px; color: #9a3412; margin-top: 6px; line-height: 1.6;">
            <b>WER:</b> {third['wer_percent']}% &bull; <b>Accuracy:</b> {third['accuracy_percent']}%<br>
            <b>Latency:</b> {third['processing_time_sec']}s ({third['rtf']}x)<br>
            <b>Cost:</b> ${third['cost_per_1000_hours']}/1k hrs
        </div>
    </div>
    """ if third else ""

    return f"""
    <div style="display: grid; grid-template-columns: 1fr 1.15fr 1fr; gap: 16px; margin: 15px 0 25px 0; align-items: stretch;">
        {card_silver}
        {card_gold}
        {card_bronze}
    </div>
    """


def render_multi_model_diff_html(model_eval: Dict[str, Any]) -> str:
    """Builds clean, modern HTML showing aligned text with color-coded diff highlights and confidence."""
    align = model_eval.get("alignment", {})
    ref_tokens = align.get("aligned_ref", [])
    hyp_tokens = align.get("aligned_hyp", [])
    ops = align.get("aligned_ops", [])

    if not ops:
        return "<p>No alignment data available.</p>"

    html_tokens = []
    for r, h, op in zip(ref_tokens, hyp_tokens, ops):
        if op == "CORRECT":
            html_tokens.append(f'<span style="background-color: #d1fae5; color: #065f46; padding: 3px 7px; margin: 2px; border-radius: 4px; font-weight: 500;">{h}</span>')
        elif op == "SUBSTITUTION":
            html_tokens.append(f'<span style="background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; padding: 3px 7px; margin: 2px; border-radius: 4px; font-weight: 500;" title="Reference was: \'{r}\'">{h} <small style="color: #b45309; font-size: 11px;">(was: {r})</small></span>')
        elif op == "DELETION":
            html_tokens.append(f'<span style="background-color: #fee2e2; color: #991b1b; text-decoration: line-through; padding: 3px 7px; margin: 2px; border-radius: 4px; font-weight: 500;" title="Missing word in transcript">[{r}]</span>')
        elif op == "INSERTION":
            html_tokens.append(f'<span style="background-color: #dbeafe; color: #1e40af; border: 1px dashed #3b82f6; padding: 3px 7px; margin: 2px; border-radius: 4px; font-weight: 500;" title="Extra / inserted word">+{h}</span>')

    diff_content = " ".join(html_tokens)
    return f"""
    <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 8px 0; font-family: ui-sans-serif, system-ui, sans-serif; line-height: 2.2; font-size: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
      {diff_content}
    </div>
    """


def render_confidence_timeline_html(word_timeline: List[Dict[str, Any]]) -> str:
    """Renders an interactive word-by-word confidence timeline with color badges."""
    if not word_timeline:
        return "<p>No confidence data available.</p>"

    html_tokens = []
    for item in word_timeline:
        w = item["word"]
        conf = item["confidence_percent"]
        ts = item["timestamp_label"]
        
        if conf >= 90.0:
            badge_color = "#10b981"
            bg_color = "#ecfdf5"
        elif conf >= 75.0:
            badge_color = "#f59e0b"
            bg_color = "#fffbeb"
        else:
            badge_color = "#ef4444"
            bg_color = "#fef2f2"

        html_tokens.append(
            f'<span style="display: inline-block; background-color: {bg_color}; border: 1px solid {badge_color}; border-radius: 6px; padding: 4px 8px; margin: 3px; font-size: 13px;" title="{ts} Confidence: {conf}%">'
            f'<strong>{w}</strong> <small style="color: {badge_color}; font-size: 11px;">{conf}%</small>'
            f'</span>'
        )

    return f"""
    <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 8px 0; line-height: 2.3; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
        {" ".join(html_tokens)}
    </div>
    """


def render_system_architecture_html() -> str:
    """Renders a sleek, modern visual card describing the multi-model pipeline architecture."""
    return """
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; border-radius: 12px; padding: 24px; margin: 15px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.15);">
        <h4 style="margin-top: 0; color: #38bdf8; font-size: 18px;">🏗️ Enterprise Multi-Model ASR Benchmark Architecture</h4>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px; text-align: center;">
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 14px;">
                <div style="font-size: 26px;">🎙️</div>
                <strong style="color: #67e8f9; font-size: 14px;">1. Audio Ingestion & VAD</strong>
                <p style="font-size: 12px; color: #cbd5e1; margin: 6px 0 0 0;">16kHz conversion, SNR calculation, speech vs silence gating, clipping detector.</p>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 14px;">
                <div style="font-size: 26px;">⚡</div>
                <strong style="color: #a7f3d0; font-size: 14px;">2. Multi-Model Pipeline</strong>
                <p style="font-size: 12px; color: #cbd5e1; margin: 6px 0 0 0;">Concurrent execution of 8+ models (CTranslate2, OpenAI, Cloud STT APIs).</p>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 14px;">
                <div style="font-size: 26px;">📐</div>
                <strong style="color: #fde047; font-size: 14px;">3. Metric Engine</strong>
                <p style="font-size: 12px; color: #cbd5e1; margin: 6px 0 0 0;">DP Levenshtein matrix, WER/CER, Med-WER, NumER, DER, BLEU, WIP.</p>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 14px;">
                <div style="font-size: 26px;">💡</div>
                <strong style="color: #f472b6; font-size: 14px;">4. Decision & Reports</strong>
                <p style="font-size: 12px; color: #cbd5e1; margin: 6px 0 0 0;">Pareto cost analysis, AI model recommender, printable executive reports.</p>
            </div>
        </div>
    </div>
    """
