# 🎙️ Enterprise Multi-Model ASR Benchmark & Audio KPI Evaluation Platform

An enterprise-grade **Streamlit** platform in Python to benchmark, profile, and evaluate **all leading Speech-to-Text (ASR) models** simultaneously against **Ground Truth Transcripts**, **Acoustic Quality**, **Domain Terminology**, **Multi-Speaker Diarization**, and **Cost Constraints**.

---

## 🌟 Comprehensive 30 Feature Suite

1. **Model Performance Dashboard**: Parallel benchmarking of 8+ models (Faster-Whisper Base/Small, OpenAI Whisper Base/Small, Google STT, Azure Speech, Deepgram Nova-2, AWS Transcribe).
2. **Audio Quality Analysis**: Signal-to-Noise Ratio (SNR dB), Audio Clipping detection, RMS energy, Dynamic Range (dB), and Spectral Centroid/Brightness.
3. **Noise Robustness Testing**: Dynamic acoustic stress testing with SNR degradation curves across Cafe, Street, and Telephony noise.
4. **Accent & Speech Speed Analysis**: Words Per Minute (WPM), Syllables per second, and conversational pace rating (Slow/Normal/Fast).
5. **Word-Level Confidence Analysis**: Word confidence probability distributions and low-confidence token highlighting (<80%, <50%).
6. **Timestamped Transcript**: High-precision word-level & segment-level start/end timestamps `[MM:SS.ms]`.
7. **Audio Waveform Synchronization**: Interactive Plotly waveform with amplitude envelope and time synchronization.
8. **Error Heatmap**: 2D matrix heatmap (Models $\times$ Error Categories: Substitutions, Deletions, Insertions, Missed Entities, Missed Numbers).
9. **Error Pattern Analysis**: Token-level alignment and deep decomposition into phonetic confusions and structural errors.
10. **Domain-wise Benchmarking**: 6 preloaded enterprise scenarios (Medical, Finance, Legal, AI/Tech, Call Center Telephony, Multilingual Translation) with domain-specific metrics (Med-WER / EER, NumER).
11. **Cost vs Accuracy Analysis**: Real-world ASR pricing models (\$ / 1,000 audio hours) vs Word Accuracy Pareto Frontier.
12. **Real-Time Microphone Transcription**: In-browser live microphone voice recording with instant multi-model benchmarking.
13. **AI Model Recommendation Engine**: Multi-criteria decision engine providing persona-based recommendations (Champion, Mission-Critical, Real-Time Streaming, Budget).
14. **Custom Performance Weighting**: Interactive sliders for Accuracy, Latency, Cost, Semantic Similarity, and Domain Entities to dynamically calculate custom Champion Scores.
15. **Benchmark History Tracking**: Session log tracking of benchmark runs with trend analysis over time.
16. **Experiment Comparison**: Side-by-side metric comparison between different benchmark experiments.
17. **Automated PDF / HTML Report Generation**: Downloadable standalone printable executive HTML/PDF report with scorecards, radar charts, and AI diagnostics.
18. **Dark / Light Mode**: Seamless theme styling toggle directly accessible in the UI.
19. **System Architecture Visualization**: Interactive visual architecture card of the end-to-end multi-model ASR evaluation pipeline.
20. **AI Results Assistant**: Natural language diagnostic engine explaining failure root causes, acoustic hurdles, and prompt-tuning suggestions.
21. **Multi-Model Parallel Transcription**: Concurrent execution across multiple ASR backends.
22. **Punctuation & Capitalization Analysis**: Punctuation retention ratio and raw vs normalized WER penalty.
23. **Speaker Diarization**: Multi-speaker dialogue segmentation (Doctor vs Patient / Agent vs Customer), Diarization Error Rate (DER %), and speaker-specific WER.
24. **Language Detection**: Spoken language identification (LID) and probability confidence scoring.
25. **Multilingual Transcription**: Multi-language evaluation across Spanish, French, German, Hindi, and English.
26. **Translation Accuracy Evaluation**: Cross-lingual speech translation benchmarking with BLEU-1, BLEU-2, and semantic similarity.
27. **Audio Silence & Speech Detection**: Energy-based Voice Activity Detection (VAD), silence ratio, and speech-to-pause ratio.
28. **Real-Time Latency Monitoring**: Processing time, Real-Time Factor (RTF), and throughput (words/sec).
29. **Model Failure/Error Detection**: Automated detection of ASR repetition loops, empty outputs, and silent dropouts.
30. **Interactive Transcript Diff Viewer**: Multi-model synchronized token diff viewer with color badges (🟢 Correct, 🟠 Substitution, 🔴 Deletion, 🔵 Insertion).

---

## 📁 Project Architecture

```
nlp ese/
├── app.py                     # Main Streamlit dashboard UI with 8 tabs & theme toggle
├── evaluator.py               # Multi-model evaluation engine (WER, CER, EER, NumER, DER, BLEU)
├── audio_analyzer.py          # Acoustic analyzer (SNR dB, clipping, VAD silence, WPM pace, noise curves)
├── transcription_engine.py    # Multi-model parallel ASR pipeline (8 backends), live mic, failure checks
├── cost_recommender.py        # Cost vs accuracy Pareto, AI recommender & AI diagnostic assistant
├── visualizer.py              # Visualizations (Multi-bar, Radar, 2D Heatmap, Waveforms, Multi-Diff)
├── report_generator.py        # Printable Executive HTML/PDF report builder & CSV/JSON exporter
├── sample_data/
│   ├── __init__.py
│   └── samples.py             # 6 domain datasets + multi-speaker + multilingual ground truths
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
