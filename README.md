# 🎙️ Enterprise Class-wise STT Benchmarking System

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive, parallelized benchmarking platform designed to evaluate and compare 8+ Speech-to-Text (ASR) architectures across specific domain classes (e.g., Healthcare, Technology, Legal). 

Instead of just looking at overall accuracy, this system provides **Class-wise Domain-Specific** insights, dynamically revealing which AI models perform best on specific types of content, vocabularies, and acoustic environments.

---

## ✨ Key Features

- **📂 Class-wise Dataset Management**: Organize your audio files into specific domains (Classes). Upload multiple audio files (`.wav, .mp3, .m4a, .mp4`, etc.) and pair them with ground-truth reference transcripts.
- **🚀 Parallel Execution Engine**: Run 8+ STT models concurrently utilizing thread-pooling for massive speedups. 
- **📊 Advanced ASR Metrics**: Calculates clamped, production-grade metrics including **WER, CER, BLEU, ROUGE-L, and METEOR**.
- **🏥 Domain Jargon Analysis**: Specialized evaluation for Entity Error Rate (EER) and Numeric Error Rate (NumER) to see how models handle specialized vocabulary.
- **🔊 Acoustic & VAD Profiling**: Evaluates background noise (SNR dB), audio clipping, and Voice Activity Detection (speech-to-silence ratios).
- **📉 Interactive Dashboards**: Beautiful Plotly-powered visualizations including 2D Error Heatmaps, Radar Charts, and Cost vs. Accuracy Pareto Frontiers.
- **🤖 Automated AI Analysis**: The system dynamically generates an analytical summary detailing which model won which class and why, based on actual benchmark results.
- **📜 CSV Executive Reporting**: Export all flattened metrics across all files, domains, and models into a consolidated CSV scorecard.

---

## 🤖 Supported STT Models

The platform evaluates the following models (with built-in simulation fallbacks for missing API keys):
1. **Faster-Whisper (Base)**
2. **Faster-Whisper (Small)**
3. **OpenAI Whisper (Base)**
4. **OpenAI Whisper (Small)**
5. **Google Speech-to-Text**
6. **Microsoft Azure Speech**
7. **Deepgram Nova-2**
8. **AWS Transcribe**

---

## ⚙️ The 8-Step Workflow

The Streamlit dashboard is structured around an intuitive 8-step pipeline:
1. **Dataset & Classes**: Define domains and upload audio + reference pairs.
2. **Dataset Review**: Verify your assembled multi-class dataset.
3. **Select & Run**: Choose your target STT models and trigger the batch execution loop.
4. **File Results**: Drill down into individual file performance, metrics, and leaderboards.
5. **Class Comparison**: Visually compare average model performance grouped by domain class.
6. **Error Analysis**: Inspect word-by-word diffs (Insertions, Deletions, Substitutions).
7. **Final Analysis**: Read the auto-generated dynamic conclusion of the benchmark.
8. **Report**: Download the final CSV scorecard for stakeholders.

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/stt-benchmark-system.git
cd stt-benchmark-system
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
*Make sure your `requirements.txt` includes: `streamlit, pandas, plotly, numpy, SpeechRecognition, openai-whisper, scipy, pydub, nltk, rouge-score, jiwer`.*

### 4. NLTK Setup (One-time)
The system uses NLTK for METEOR score calculations. The app handles this automatically, but you can pre-download the WordNet corpus by running:
```python
python -c "import nltk; nltk.download('wordnet')"
```

---

## 💻 Usage

Start the Streamlit application by running:

```bash
streamlit run app.py
```

The web dashboard will automatically open in your default browser at `http://localhost:8501`. 

---

## 📁 Project Structure

```text
stt-benchmark-system/
│
├── app.py                     # Main Streamlit application & UI workflow
├── evaluator.py               # Core metrics engine (WER, CER, ROUGE, METEOR)
├── transcription_engine.py    # Parallel STT execution & audio formatting
├── visualizer.py              # Plotly chart generation & HTML rendering
├── audio_analyzer.py          # SNR, VAD, and acoustic quality analysis
├── cost_recommender.py        # Pareto frontier & cost efficiency logic
├── report_generator.py        # PDF/HTML export generation
├── sample_data/               # Directory for built-in sample audio generation
└── README.md                  # Project documentation
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/stt-benchmark-system/issues).

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
