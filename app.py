"""
app.py
------
Enterprise Multi-Model Speech-to-Text Comparison & Audio Evaluation Platform.
Benchmarking 8+ ASR Architectures simultaneously against Ground Truth.
Now with Class-wise Domain-Specific Benchmarking System.
"""

import os
import json
import time
import tempfile
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Local module imports
from evaluator import (
    evaluate_all_models,
    evaluate_single_transcription,
    normalize_text
)
from transcription_engine import (
    AVAILABLE_MODELS,
    run_parallel_multi_model_transcriptions,
    get_audio_duration,
    convert_to_wav
)
from audio_analyzer import (
    analyze_audio_quality,
    analyze_voice_activity_and_silence,
    analyze_speech_pace_and_speed
)
from cost_recommender import (
    calculate_cost_accuracy_pareto,
    generate_ai_model_recommendations,
    generate_ai_diagnostic_summary
)
from visualizer import (
    plot_all_models_kpi_bars,
    plot_multi_model_radar,
    plot_error_heatmap_2d,
    plot_cost_vs_accuracy_pareto,
    plot_noise_robustness_curves,
    plot_interactive_audio_waveform,
    plot_snr_gauge_meter,
    plot_speech_silence_donut,
    render_champion_podium_html,
    render_multi_model_diff_html,
    render_confidence_timeline_html,
    render_system_architecture_html,
    plot_class_comparison_bars
)
from report_generator import generate_executive_html_report
from sample_data.samples import SAMPLE_DATASETS, generate_sample_audio_wav


# Page Configuration
st.set_page_config(
    page_title="Class-wise Domain-Specific STT Benchmark",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize Hierarchical Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"
if "selected_models_state" not in st.session_state:
    st.session_state["selected_models_state"] = AVAILABLE_MODELS[:4]

# --- NEW HIERARCHICAL STATE ---
# dataset structure: Dict[str, List[Dict]]
# e.g. { "Educational": [ {"file_name": "test.wav", "audio_path": "/tmp/test.wav", "reference": "abc"} ] }
if "dataset" not in st.session_state:
    st.session_state["dataset"] = {} 
if "class_results" not in st.session_state:
    st.session_state["class_results"] = {}


def inject_custom_css(theme: str):
    """Injects responsive custom styling based on theme."""
    is_dark = theme == "Dark"
    bg_card = "#1e293b" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#1e293b"
    border_color = "#334155" if is_dark else "#e2e8f0"
    hero_bg = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)" if is_dark else "linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%)"

    st.markdown(f"""
    <style>
        .stApp {{
            color: {text_color};
        }}
        .hero-banner {{
            background: {hero_bg};
            color: white;
            padding: 24px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }}
        .step-pill {{
            background: rgba(255,255,255,0.15);
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
            margin-right: 8px;
        }}
        .metric-card {{
            background-color: {bg_card};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }}
        .domain-badge {{
            background-color: #dbeafe;
            color: #1e40af;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            display: inline-block;
            margin-right: 6px;
        }}
    </style>
    """, unsafe_allow_html=True)


def run_benchmark_batch(dataset: Dict[str, List[Dict[str, Any]]], models: List[str], weights: Dict[str, float]):
    """Encapsulates execution of the multi-model benchmark across all classes and files."""
    results = {}
    total_files = sum(len(files) for files in dataset.values())
    if total_files == 0:
        return results

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current = 0
    for cls_name, files in dataset.items():
        results[cls_name] = []
        for file_data in files:
            audio_path = file_data["audio_path"]
            ground_truth = file_data["reference"]
            file_name = file_data["file_name"]
            
            status_text.text(f"Processing Class: {cls_name} | File: {file_name}...")
            audio_dur = get_audio_duration(audio_path)
            
            # Parallel Transcription
            models_outputs = run_parallel_multi_model_transcriptions(
                audio_path=audio_path,
                selected_models=models,
                ground_truth=ground_truth,
                is_synthetic_benchmark=False
            )
            
            # KPI Evaluation
            eval_results = evaluate_all_models(
                reference=ground_truth,
                models_hypotheses=models_outputs,
                audio_duration_sec=audio_dur,
                reference_translation="",
                weights=weights
            )
            
            # Get VAD and audio quality
            aq = analyze_audio_quality(audio_path)
            vad = analyze_voice_activity_and_silence(audio_path)
            
            results[cls_name].append({
                "file_name": file_name,
                "audio_path": audio_path,
                "reference": ground_truth,
                "eval_results": eval_results,
                "audio_quality": aq,
                "vad_results": vad
            })
            current += 1
            progress_bar.progress(current / total_files)
            
    status_text.text("Batch processing complete!")
    return results

def generate_final_analysis_text(class_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Automatically generates a meaningful analytical summary based on actual results."""
    if not class_results:
        return "No results available to analyze."
        
    class_winners = {}
    class_wer = {}
    model_wins = {}
    
    # Calculate best models per class based on average custom champion score
    for cls_name, files_data in class_results.items():
        model_scores = {}
        model_wers = {}
        for f_data in files_data:
            if "eval_results" in f_data and f_data["eval_results"]:
                for m in f_data["eval_results"]["models"]:
                    m_name = m["algorithm_name"]
                    if m_name not in model_scores:
                        model_scores[m_name] = []
                        model_wers[m_name] = []
                    model_scores[m_name].append(m["custom_champion_score"])
                    model_wers[m_name].append(m["wer_percent"])
                    
        avg_scores = {m: np.mean(scores) for m, scores in model_scores.items()}
        avg_wers = {m: np.mean(wers) for m, wers in model_wers.items()}
        
        if avg_scores:
            best_model = max(avg_scores, key=avg_scores.get)
            class_winners[cls_name] = best_model
            class_wer[cls_name] = avg_wers[best_model]
            model_wins[best_model] = model_wins.get(best_model, 0) + 1

    if not class_winners:
        return "Unable to generate analysis (no evaluation data)."

    analysis_lines = []
    analysis_lines.append("### 🤖 Dynamic Benchmark Analysis\n")
    
    # Domain-specific performance
    for cls_name, winner in class_winners.items():
        analysis_lines.append(f"- **{winner}** performed best in the **{cls_name}** class with an average WER of **{class_wer[cls_name]:.2f}%**.")
        
    # Overall Consistency
    if model_wins:
        overall_best = max(model_wins, key=model_wins.get)
        analysis_lines.append(f"\n**Overall Consistency:** **{overall_best}** was the most consistent model, winning {model_wins[overall_best]} out of {len(class_winners)} classes. ")
        if len(class_winners) > 1:
            analysis_lines.append("This suggests that STT performance " + 
                                  ("varies according to domain characteristics, requiring specialized model routing." if len(model_wins) > 1 else "is highly robust across all tested domains with a clear single winner."))
    
    return "\n".join(analysis_lines)


def main():
    with st.sidebar:
        st.markdown("### ⚙️ Platform Settings")
        theme_choice = st.radio("UI Theme Mode:", ["Light", "Dark"], horizontal=True)
        st.session_state["theme_mode"] = theme_choice
        inject_custom_css(theme_choice)

        st.markdown("---")
        st.markdown("### 🎚️ Performance Ranking Weights")
        st.caption("Adjust priority weights to customize the Champion score calculation:")
        w_acc = st.slider("Word Accuracy (WER/CER):", 0.0, 1.0, 0.40, 0.05)
        w_lat = st.slider("Latency & Speed (RTF):", 0.0, 1.0, 0.25, 0.05)
        w_cost = st.slider("Cost Efficiency ($):", 0.0, 1.0, 0.15, 0.05)
        w_sim = st.slider("Semantic Similarity:", 0.0, 1.0, 0.10, 0.05)
        w_ent = st.slider("Domain Jargon Accuracy:", 0.0, 1.0, 0.10, 0.05)

        total_w = w_acc + w_lat + w_cost + w_sim + w_ent
        norm_weights = {
            "accuracy": w_acc / max(total_w, 0.01),
            "latency": w_lat / max(total_w, 0.01),
            "cost": w_cost / max(total_w, 0.01),
            "similarity": w_sim / max(total_w, 0.01),
            "entities": w_ent / max(total_w, 0.01)
        }

    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <h1 style="margin: 0; font-size: 28px; color: white;">🎙️ Class-wise Domain-Specific STT Benchmark</h1>
                <p style="margin: 6px 0 12px 0; opacity: 0.95; font-size: 15px;">
                    Organize your audio into domain classes, assign references, and batch evaluate across 8+ STT models.
                </p>
                <div>
                    <span class="step-pill">1. Build Dataset</span>
                    <span class="step-pill">2. Batch Run Models</span>
                    <span class="step-pill">3. Class-wise Analysis</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📁 1. Dataset & Classes",
        "🎧 2. Audio/Video + Refs",
        "🚀 3. Select & Run",
        "📄 4. File Results",
        "📊 5. Class Comparison",
        "🔍 6. Error Analysis",
        "🤖 7. Final Analysis",
        "📜 8. Report"
    ])

    # ==========================================
    # TAB 1: DATASET & CLASSES
    # ==========================================
    with tab1:
        st.markdown("## 📁 Define Domain Classes & Upload Files")
        st.caption("Create multiple domain classes. Add ONE verified reference transcript for EVERY audio file.")
        
        # Add new class
        col_c, col_a = st.columns([1, 1])
        with col_c:
            new_class_name = st.text_input("Create New Class / Domain (e.g., Educational, Sports):")
            if st.button("➕ Add Class"):
                if new_class_name and new_class_name not in st.session_state["dataset"]:
                    st.session_state["dataset"][new_class_name] = []
                    st.success(f"Class '{new_class_name}' added!")
        
        st.markdown("---")
        if not st.session_state["dataset"]:
            st.info("No classes defined yet. Add a class above.")
        else:
            for cls_name, files in st.session_state["dataset"].items():
                st.markdown(f"### 🏷️ Class: **{cls_name}**")
                st.markdown(f"*Files ready: {len(files)}*")
                
                with st.expander(f"➕ Add Audio to '{cls_name}'"):
                    upl = st.file_uploader(f"Upload media for {cls_name}:", type=["wav", "mp3", "m4a", "mp4", "mov", "avi", "mkv"], key=f"up_{cls_name}")
                    ref_text = st.text_area(f"Reference Transcript for this file:", key=f"ref_{cls_name}")
                    if st.button(f"Add to {cls_name}", key=f"btn_{cls_name}"):
                        if upl and ref_text.strip():
                            temp_dir = tempfile.gettempdir()
                            file_path = os.path.join(temp_dir, upl.name)
                            with open(file_path, "wb") as f:
                                f.write(upl.getbuffer())
                            
                            st.session_state["dataset"][cls_name].append({
                                "file_name": upl.name,
                                "audio_path": file_path,
                                "reference": ref_text.strip()
                            })
                            st.success(f"Added {upl.name} to {cls_name}!")
                            st.rerun()
                        else:
                            st.error("Upload a file AND provide a reference transcript.")

    # ==========================================
    # TAB 2: AUDIO / REFS REVIEW
    # ==========================================
    with tab2:
        st.markdown("## 🎧 Dataset Review")
        total_files = sum(len(f) for f in st.session_state["dataset"].values())
        st.metric("Total Benchmark Ready Files", total_files)
        
        for cls_name, files in st.session_state["dataset"].items():
            st.markdown(f"#### 🏷️ {cls_name}")
            if not files:
                st.caption("No files.")
            for f in files:
                st.markdown(f"**{f['file_name']}**")
                st.audio(f['audio_path'])
                st.info(f['reference'])
            st.markdown("---")

    # ==========================================
    # TAB 3: SELECT & RUN
    # ==========================================
    with tab3:
        st.markdown("## 🚀 Batch Run Multi-Model Benchmark")
        st.caption("Run all selected models across every file in every class simultaneously.")
        
        selected_models = st.multiselect(
            "Active STT Models:",
            AVAILABLE_MODELS,
            default=st.session_state.get("selected_models_state", AVAILABLE_MODELS[:4])
        )
        st.session_state["selected_models_state"] = selected_models
        
        if st.button("🔥 Run Batch Benchmark", type="primary", width="stretch"):
            if not st.session_state["dataset"]:
                st.error("Dataset is empty! Add files in Tab 1.")
            elif not selected_models:
                st.error("Select at least one model!")
            else:
                with st.spinner("Executing Batch Processing..."):
                    res = run_benchmark_batch(st.session_state["dataset"], selected_models, norm_weights)
                    st.session_state["class_results"] = res
                st.success("Benchmark Complete! Check File Results and Class Comparison Tabs.")

    # ==========================================
    # TAB 4: FILE RESULTS
    # ==========================================
    with tab4:
        st.markdown("## 📄 Individual File Metrics (WER/CER/BLEU/ROUGE/METEOR)")
        if not st.session_state["class_results"]:
            st.info("Run the benchmark in Tab 3 first.")
        else:
            cls_sel = st.selectbox("Select Class:", list(st.session_state["class_results"].keys()), key="t4_c")
            if cls_sel and st.session_state["class_results"][cls_sel]:
                file_opts = [f["file_name"] for f in st.session_state["class_results"][cls_sel]]
                file_sel = st.selectbox("Select File:", file_opts, key="t4_f")
                
                # Find file
                f_data = next((f for f in st.session_state["class_results"][cls_sel] if f["file_name"] == file_sel), None)
                if f_data and "eval_results" in f_data:
                    eval_results = f_data["eval_results"]
                    st.markdown("### 🏆 Leaderboard for this File")
                    
                    lb_rows = []
                    for rank, m in enumerate(eval_results["leaderboard"], 1):
                        lb_rows.append({
                            "Rank": f"#{rank}",
                            "Model": m["algorithm_name"],
                            "WER": f"{m['wer_percent']}%",
                            "CER": f"{m['cer_percent']}%",
                            "ROUGE-L": f"{m.get('rouge_evaluation',{}).get('rougeL_fmeasure_percent', 0.0)}%",
                            "METEOR": f"{m.get('meteor_evaluation',{}).get('meteor_score_percent', 0.0)}%",
                            "BLEU-1": f"{m.get('translation_evaluation',{}).get('bleu_1', 0.0)}%",
                            "Time (s)": m['processing_time_sec']
                        })
                    st.dataframe(pd.DataFrame(lb_rows), width="stretch", hide_index=True)
                    
                    st.plotly_chart(plot_all_models_kpi_bars(eval_results), width="stretch")
                    st.plotly_chart(plot_multi_model_radar(eval_results), width="stretch")

    # ==========================================
    # TAB 5: CLASS COMPARISON
    # ==========================================
    with tab5:
        st.markdown("## 📊 Class-wise Domain Comparison")
        if not st.session_state["class_results"]:
            st.info("Run the benchmark first.")
        else:
            m_choice = st.selectbox("Select Metric for Cross-Class Comparison:", [
                "wer_percent", "accuracy_percent", "rouge_evaluation.rougeL_fmeasure_percent", "meteor_evaluation.meteor_score_percent"
            ])
            st.plotly_chart(plot_class_comparison_bars(st.session_state["class_results"], metric=m_choice), width="stretch")

    # ==========================================
    # TAB 6: ERROR ANALYSIS
    # ==========================================
    with tab6:
        st.markdown("## 🔍 Transcript & Error Analysis")
        st.caption("Compare reference vs generated transcript (Insertions, Deletions, Substitutions)")
        if not st.session_state["class_results"]:
            st.info("Run the benchmark first.")
        else:
            c6_c = st.selectbox("Class:", list(st.session_state["class_results"].keys()), key="t6_c")
            if c6_c and st.session_state["class_results"][c6_c]:
                f_opts = [f["file_name"] for f in st.session_state["class_results"][c6_c]]
                c6_f = st.selectbox("File:", f_opts, key="t6_f")
                f_data = next((f for f in st.session_state["class_results"][c6_c] if f["file_name"] == c6_f), None)
                
                if f_data and "eval_results" in f_data:
                    eval_results = f_data["eval_results"]
                    st.plotly_chart(plot_error_heatmap_2d(eval_results), width="stretch")
                    
                    st.markdown("### 📝 Word-by-Word Diff Inspection")
                    diff_tabs = st.tabs([m["algorithm_name"] for m in eval_results["models"]])
                    for idx, tab_item in enumerate(diff_tabs):
                        with tab_item:
                            m_eval = eval_results["models"][idx]
                            st.markdown(render_multi_model_diff_html(m_eval), unsafe_allow_html=True)
                            st.caption(
                                f"❌ Substitutions: **{m_eval['substitutions']}** | "
                                f"🚫 Missing: **{m_eval['deletions']}** | "
                                f"➕ Extra: **{m_eval['insertions']}**"
                            )

    # ==========================================
    # TAB 7: FINAL ANALYSIS
    # ==========================================
    with tab7:
        st.markdown("## 🤖 Auto-Generated Final Analysis")
        if not st.session_state["class_results"]:
            st.info("Run the benchmark first.")
        else:
            analysis_text = generate_final_analysis_text(st.session_state["class_results"])
            st.markdown(
                f"<div style='background-color: #f1f5f9; padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6;'>"
                f"{analysis_text}</div>", 
                unsafe_allow_html=True
            )

    # ==========================================
    # TAB 8: REPORT
    # ==========================================
    with tab8:
        st.markdown("## 📜 Consolidated Executive Report")
        if not st.session_state["class_results"]:
            st.info("Run the benchmark first.")
        else:
            st.success("Data is ready! Click below to download the final CSV scorecard.")
            
            # Build flat CSV
            flat_rows = []
            for cls, files in st.session_state["class_results"].items():
                for f in files:
                    if "eval_results" in f:
                        for m in f["eval_results"]["models"]:
                            flat_rows.append({
                                "Class": cls,
                                "File": f["file_name"],
                                "Model": m["algorithm_name"],
                                "WER (%)": m["wer_percent"],
                                "CER (%)": m["cer_percent"],
                                "ROUGE-L (%)": m.get('rouge_evaluation',{}).get('rougeL_fmeasure_percent', 0.0),
                                "METEOR (%)": m.get('meteor_evaluation',{}).get('meteor_score_percent', 0.0),
                                "Time (s)": m["processing_time_sec"]
                            })
            if flat_rows:
                df = pd.DataFrame(flat_rows)
                st.dataframe(df, width="stretch")
                st.download_button(
                    label="📥 Download Full Batch Metrics (CSV)",
                    data=df.to_csv(index=False),
                    file_name="class_wise_stt_benchmark.csv",
                    mime="text/csv",
                    type="primary"
                )

if __name__ == "__main__":
    main()
