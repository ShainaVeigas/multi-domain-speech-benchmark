"""
report_generator.py
--------------------
Automated Executive Benchmark Report Generator.
Builds standalone, printable HTML & PDF-ready benchmark reports with
KPI tables, champion badges, domain analysis, acoustic health scorecards,
and AI recommendations.
"""

import json
from typing import Dict, Any, List
import pandas as pd


def generate_executive_html_report(
    eval_data: Dict[str, Any],
    audio_quality: Dict[str, Any],
    ai_recommendations: List[Dict[str, Any]],
    ai_diagnostic: str
) -> str:
    """
    Constructs an executive-ready, printable HTML/PDF report with embedded styling.
    """
    winner = eval_data.get("winner_model", {})
    all_models = eval_data.get("models", [])
    gt_stats = eval_data.get("ground_truth_stats", {})

    # Build Model Rows
    model_rows_html = ""
    for idx, m in enumerate(all_models, 1):
        is_win = m["algorithm_name"] == eval_data.get("winner", "")
        row_bg = "#ecfdf5" if is_win else ("#ffffff" if idx % 2 == 0 else "#f9fafb")
        badge = " 🏆 <b>Winner</b>" if is_win else ""
        
        model_rows_html += f"""
        <tr style="background-color: {row_bg}; border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 10px 14px; font-weight: 600;">{idx}. {m['algorithm_name']}{badge}</td>
            <td style="padding: 10px 14px; text-align: center; color: #059669; font-weight: bold;">{m.get('custom_champion_score', 0)}/100</td>
            <td style="padding: 10px 14px; text-align: center;">{m['wer_percent']}%</td>
            <td style="padding: 10px 14px; text-align: center;">{m['cer_percent']}%</td>
            <td style="padding: 10px 14px; text-align: center;">{m['accuracy_percent']}%</td>
            <td style="padding: 10px 14px; text-align: center;">{m['similarity_percent']}%</td>
            <td style="padding: 10px 14px; text-align: center;">{m['processing_time_sec']}s ({m['rtf']}x)</td>
            <td style="padding: 10px 14px; text-align: center;">${m['cost_per_1000_hours']}</td>
        </tr>
        """

    # Build Recommendations HTML
    rec_cards_html = ""
    for r in ai_recommendations:
        rec_cards_html += f"""
        <div style="background-color: #f8fafc; border-left: 4px solid {r['color']}; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="font-size: 15px; color: #1e293b;">{r['persona']} &mdash; <span style="color: {r['color']};">{r['model_name']}</span></strong>
                <span style="background-color: {r['color']}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{r['badge']}</span>
            </div>
            <p style="margin: 6px 0; color: #475569; font-size: 13px;">{r['reason']}</p>
            <small style="color: #64748b; font-weight: 500;">{r['metrics']}</small>
        </div>
        """

    # HTML Template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Speech-to-Text Multi-Model Benchmark Executive Report</title>
<style>
    @media print {{
        body {{ font-size: 12px; }}
        .no-print {{ display: none; }}
        .page-break {{ page-break-before: always; }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        line-height: 1.5;
        margin: 0;
        padding: 30px;
        background-color: #f1f5f9;
    }}
    .report-container {{
        max-width: 960px;
        margin: 0 auto;
        background: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }}
    .header-banner {{
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 20px;
        margin-bottom: 25px;
    }}
    .winner-box {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }}
    .metric-card {{
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13px;
    }}
    th {{
        background-color: #1e293b;
        color: white;
        padding: 10px 14px;
        font-weight: 600;
    }}
    .section-title {{
        color: #0f172a;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        margin-top: 30px;
        margin-bottom: 16px;
        font-size: 18px;
    }}
    .print-btn {{
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: bold;
        float: right;
    }}
</style>
</head>
<body>
<div class="report-container">
    <div class="header-banner">
        <button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
        <h1 style="margin: 0; font-size: 26px; color: #0f172a;">🎙️ ASR Multi-Model Benchmark & Audio KPI Report</h1>
        <p style="margin: 5px 0 0 0; color: #64748b; font-size: 14px;">Enterprise Speech-to-Text Comparative Evaluation, Acoustic Profiling & Decision Matrix</p>
    </div>

    <!-- Champion Banner -->
    <div class="winner-box">
        <h2 style="margin: 0 0 6px 0; font-size: 22px;">🏆 Champion: {eval_data.get('winner', 'N/A')}</h2>
        <p style="margin: 0; opacity: 0.95; font-size: 14px;">{eval_data.get('winner_reason', '')}</p>
    </div>

    <!-- Acoustic Health Scorecard -->
    <h3 class="section-title">🔊 Audio Quality & Environment Health</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div style="font-size: 12px; color: #64748b;">Signal-to-Noise Ratio (SNR)</div>
            <div style="font-size: 20px; font-weight: bold; color: #059669;">{audio_quality.get('snr_db', 0)} dB</div>
            <small style="color: #475569;">{audio_quality.get('snr_grade', 'Normal')}</small>
        </div>
        <div class="metric-card">
            <div style="font-size: 12px; color: #64748b;">Audio Duration</div>
            <div style="font-size: 20px; font-weight: bold; color: #2563eb;">{audio_quality.get('duration_sec', 0)}s</div>
            <small style="color: #475569;">Sample Rate: {audio_quality.get('sample_rate', 16000)} Hz</small>
        </div>
        <div class="metric-card">
            <div style="font-size: 12px; color: #64748b;">Audio Clipping</div>
            <div style="font-size: 20px; font-weight: bold; color: {'#ef4444' if audio_quality.get('has_clipping', False) else '#10b981'};">{audio_quality.get('clipping_percent', 0)}%</div>
            <small style="color: #475569;">Peak: {audio_quality.get('peak_amplitude', 0)}</small>
        </div>
        <div class="metric-card">
            <div style="font-size: 12px; color: #64748b;">Dynamic Range</div>
            <div style="font-size: 20px; font-weight: bold; color: #8b5cf6;">{audio_quality.get('dynamic_range_db', 0)} dB</div>
            <small style="color: #475569;">RMS: {audio_quality.get('rms_energy_db', 0)} dB</small>
        </div>
    </div>

    <!-- Multi-Model Leaderboard Table -->
    <h3 class="section-title">📊 Multi-Model Performance Leaderboard</h3>
    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Champion Score</th>
                <th>WER (%)</th>
                <th>CER (%)</th>
                <th>Accuracy</th>
                <th>Similarity</th>
                <th>Latency (RTF)</th>
                <th>Cost / 1k Hrs</th>
            </tr>
        </thead>
        <tbody>
            {model_rows_html}
        </tbody>
    </table>

    <!-- AI Production Recommendations -->
    <h3 class="section-title">💡 Production Persona Recommendations</h3>
    {rec_cards_html}

    <!-- Diagnostic Insights -->
    <h3 class="section-title">🤖 AI Diagnostic Analysis</h3>
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; font-size: 13px; color: #334155;">
        <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">{ai_diagnostic}</pre>
    </div>

    <!-- Reference Ground Truth -->
    <h3 class="section-title">📝 Reference Ground-Truth Transcript</h3>
    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 4px; font-size: 13px; color: #334155;">
        {gt_stats.get('raw_text', '')}
    </div>

    <div style="margin-top: 35px; text-align: center; color: #94a3b8; font-size: 12px; border-top: 1px solid #e2e8f0; padding-top: 15px;">
        Generated by Speech-to-Text Multi-Model Benchmark & Audio KPI Evaluation Platform
    </div>
</div>
</body>
</html>
"""
    return html
