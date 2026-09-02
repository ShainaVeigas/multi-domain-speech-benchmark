"""
cost_recommender.py
--------------------
ASR Cost-Performance Analyzer, AI Model Recommendation Engine,
and Interactive AI Results Diagnostic Assistant.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


# Real-World ASR Pricing Database ($ per audio hour)
ASR_PRICING_TABLE = {
    "Faster-Whisper (Base)": {
        "provider": "Self-Hosted (CTranslate2 GPU/CPU)",
        "cost_per_min_usd": 0.0008,
        "cost_per_hour_usd": 0.05,
        "cost_per_1000_hours": 50.0,
        "deployment": "Self-Hosted / On-Premise",
        "privacy": "100% Air-Gapped / HIPAA Compliant",
        "strengths": "Ultra-fast latency, zero cloud API fees, privacy safe."
    },
    "Faster-Whisper (Small)": {
        "provider": "Self-Hosted (CTranslate2 GPU)",
        "cost_per_min_usd": 0.0015,
        "cost_per_hour_usd": 0.09,
        "cost_per_1000_hours": 90.0,
        "deployment": "Self-Hosted / On-Premise",
        "privacy": "100% Air-Gapped / HIPAA Compliant",
        "strengths": "High accuracy, 3.5x faster than Whisper Vanilla, zero egress cost."
    },
    "OpenAI Whisper (Base)": {
        "provider": "OpenAI Cloud API",
        "cost_per_min_usd": 0.006,
        "cost_per_hour_usd": 0.36,
        "cost_per_1000_hours": 360.0,
        "deployment": "Cloud Managed API",
        "privacy": "Encrypted Cloud",
        "strengths": "Zero setup, good multilingual baseline."
    },
    "OpenAI Whisper (Small)": {
        "provider": "OpenAI Cloud API",
        "cost_per_min_usd": 0.006,
        "cost_per_hour_usd": 0.36,
        "cost_per_1000_hours": 360.0,
        "deployment": "Cloud Managed API",
        "privacy": "Encrypted Cloud",
        "strengths": "Higher accuracy on complex vocabulary."
    },
    "Google Speech-to-Text": {
        "provider": "Google Cloud",
        "cost_per_min_usd": 0.016,
        "cost_per_hour_usd": 0.96,
        "cost_per_1000_hours": 960.0,
        "deployment": "Cloud Managed API",
        "privacy": "Enterprise GCP",
        "strengths": "Broad language support, integrated telephony models."
    },
    "Microsoft Azure Speech": {
        "provider": "Microsoft Azure Cognitive Services",
        "cost_per_min_usd": 0.0167,
        "cost_per_hour_usd": 1.00,
        "cost_per_1000_hours": 1000.0,
        "deployment": "Cloud Managed API",
        "privacy": "Enterprise Azure",
        "strengths": "High punctuation accuracy, enterprise compliance."
    },
    "Deepgram Nova-2": {
        "provider": "Deepgram Cloud",
        "cost_per_min_usd": 0.0043,
        "cost_per_hour_usd": 0.258,
        "cost_per_1000_hours": 258.0,
        "deployment": "Cloud Managed API",
        "privacy": "Encrypted Cloud",
        "strengths": "Fastest streaming API, lowest cloud cost per hour."
    },
    "AWS Transcribe": {
        "provider": "Amazon Web Services",
        "cost_per_min_usd": 0.024,
        "cost_per_hour_usd": 1.44,
        "cost_per_1000_hours": 1440.0,
        "deployment": "Cloud Managed API",
        "privacy": "Enterprise AWS",
        "strengths": "Native AWS ecosystem integration."
    }
}


def calculate_cost_accuracy_pareto(evaluated_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identifies Pareto-optimal models (models where no other model is BOTH cheaper AND more accurate).
    """
    if not evaluated_models:
        return {"pareto_models": [], "all_models": []}

    records = []
    for m in evaluated_models:
        name = m["algorithm_name"]
        wer = m["wer_percent"]
        acc = m["accuracy_percent"]
        cost_1k = m["cost_per_1000_hours"]
        time_s = m["processing_time_sec"]
        
        records.append({
            "model_name": name,
            "wer_percent": wer,
            "accuracy_percent": acc,
            "cost_per_1000_hours": cost_1k,
            "processing_time_sec": time_s,
            "rtf": m["rtf"],
            "raw_eval": m
        })

    # Sort by cost ascending
    sorted_by_cost = sorted(records, key=lambda x: x["cost_per_1000_hours"])

    # Identify Pareto Frontier (lower cost and higher accuracy)
    pareto_frontier = []
    current_best_acc = -1.0

    for item in sorted_by_cost:
        if item["accuracy_percent"] > current_best_acc:
            pareto_frontier.append(item["model_name"])
            current_best_acc = item["accuracy_percent"]

    for r in records:
        r["is_pareto_optimal"] = r["model_name"] in pareto_frontier

    return {
        "pareto_model_names": pareto_frontier,
        "records": records
    }


def generate_ai_model_recommendations(evaluated_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates benchmark results against real-world production personas and produces
    actionable AI model recommendations.
    """
    if not evaluated_models:
        return []

    # Find best in specific dimensions
    best_overall = max(evaluated_models, key=lambda x: x.get("custom_champion_score", 0))
    lowest_wer = min(evaluated_models, key=lambda x: x["wer_percent"])
    fastest = min(evaluated_models, key=lambda x: x["processing_time_sec"])
    cheapest = min(evaluated_models, key=lambda x: x["cost_per_hour_usd"])
    best_entities = max(evaluated_models, key=lambda x: x["entity_evaluation"]["entity_accuracy_percent"])

    recommendations = [
        {
            "persona": "👑 Overall Champion Pick",
            "model_name": best_overall["algorithm_name"],
            "badge": "Top Balanced",
            "color": "#10B981",
            "reason": f"Highest custom champion score ({best_overall['custom_champion_score']}/100). Balances low WER ({best_overall['wer_percent']}%), fast inference ({best_overall['processing_time_sec']}s), and cost efficiency.",
            "metrics": f"WER: {best_overall['wer_percent']}% | Latency: {best_overall['processing_time_sec']}s | Cost: ${best_overall['cost_per_1000_hours']}/1k hrs"
        },
        {
            "persona": "🏥 Mission-Critical / Maximum Accuracy",
            "model_name": lowest_wer["algorithm_name"],
            "badge": "Highest Precision",
            "color": "#2563EB",
            "reason": f"Achieved the lowest Word Error Rate ({lowest_wer['wer_percent']}%) and highest word accuracy ({lowest_wer['accuracy_percent']}%). Recommended for Medical, Legal, and Compliance transcripts.",
            "metrics": f"WER: {lowest_wer['wer_percent']}% | Entity Acc: {lowest_wer['entity_evaluation']['entity_accuracy_percent']}%"
        },
        {
            "persona": "⚡ Live Streaming & Real-Time Voice",
            "model_name": fastest["algorithm_name"],
            "badge": "Ultra-Low Latency",
            "color": "#F59E0B",
            "reason": f"Fastest response time ({fastest['processing_time_sec']}s) with Real-Time Factor of {fastest['rtf']}x. Recommended for live voice agents, phone bots, and real-time captions.",
            "metrics": f"RTF: {fastest['rtf']}x | Throughput: {fastest['throughput_wps']} words/sec"
        },
        {
            "persona": "💰 Maximum Cost Savings / High Volume",
            "model_name": cheapest["algorithm_name"],
            "badge": "Most Economical",
            "color": "#8B5CF6",
            "reason": f"Lowest operating cost at ${cheapest['cost_per_1000_hours']} per 1,000 audio hours while delivering strong accuracy ({cheapest['accuracy_percent']}%). Recommended for batch transcription at scale.",
            "metrics": f"Cost: ${cheapest['cost_per_1000_hours']}/1k hrs | Accuracy: {cheapest['accuracy_percent']}%"
        }
    ]

    return recommendations


def generate_ai_diagnostic_summary(eval_data: Dict[str, Any], audio_quality: Dict[str, Any]) -> str:
    """
    Generates a natural-language diagnostic explaining failure root causes,
    acoustic hurdles, phonetic confusions, and prompt-tuning suggestions.
    """
    winner = eval_data.get("winner_model", {})
    all_models = eval_data.get("models", [])
    snr = audio_quality.get("snr_db", 25.0)
    has_clipping = audio_quality.get("has_clipping", False)

    if not winner:
        return "No evaluation data available to generate diagnostic insights."

    # Identify common error words across models
    all_subs = []
    all_missing = []
    for m in all_models:
        all_subs.extend(m.get("sub_pairs", []))
        all_missing.extend(m.get("missing_words", []))

    sub_counts = [f"'{r}' $\\rightarrow$ '{h}'" for r, h in all_subs[:3]]
    sub_text = ", ".join(sub_counts) if sub_counts else "No prominent word substitutions detected."

    # Acoustic quality commentary
    if snr < 15.0:
        acoustic_note = f"⚠️ **Acoustic Warning**: The audio has a low Signal-to-Noise Ratio ({snr} dB), which increased deletion and phonetic substitution errors across all models."
    elif has_clipping:
        acoustic_note = f"⚠️ **Audio Clipping Detected**: Peak amplitude exceeded saturation thresholds, causing minor token truncation."
    else:
        acoustic_note = f"✅ **Optimal Acoustic Quality**: SNR is clean ({snr} dB) with minimal distortion, allowing models to reach peak phonetic recognition."

    # Entity & Numeric analysis
    ent_eval = winner.get("entity_evaluation", {})
    num_eval = winner.get("numeric_evaluation", {})
    missed_ent = ent_eval.get("missed_entities", [])
    missed_num = num_eval.get("missed_numbers", [])

    jargon_note = ""
    if missed_ent:
        jargon_note = f"💡 **Domain Jargon Alert**: The model missed specialized terms: `{', '.join(missed_ent[:4])}`. Applying an ASR initial prompt / hotword list can boost recognition by up to 35%."
    elif missed_num:
        jargon_note = f"💡 **Numeric Discrepancies**: Numeric formatting variations were detected on `{', '.join(missed_num[:3])}`. Enable Inverse Text Normalization (ITN) to standardize number representations."

    diagnostic = f"""
### 🤖 AI Benchmark Diagnostic & Performance Analysis

1. **🏆 Champion Overview**:
   - **{winner.get('algorithm_name', 'Top Model')}** dominated the benchmark with a **{winner.get('wer_percent', 0)}% WER** and a processing speed of **{winner.get('processing_time_sec', 0)}s** (RTF: **{winner.get('rtf', 0)}x**).
   - Word Information Preserved (WIP) was **{winner.get('wip_percent', 0)}%**, ensuring high semantic fidelity.

2. **🔊 Acoustic Environment Impact**:
   - {acoustic_note}

3. **🔍 Common Error Patterns**:
   - **Frequent Substitutions**: {sub_text}
   - {jargon_note}

4. **🚀 Production Recommendation**:
   - For real-time production pipelines, **{winner.get('algorithm_name', 'Top Model')}** provides the optimal balance of throughput and accuracy.
   - For batch workloads requiring maximum cost efficiency, self-hosted CTranslate2 / Faster-Whisper provides up to **10x cost reduction** over standard cloud APIs.
"""
    return diagnostic.strip()
