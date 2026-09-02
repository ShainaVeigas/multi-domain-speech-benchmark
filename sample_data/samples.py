"""
sample_data/samples.py
----------------------
Comprehensive pre-loaded benchmark datasets across 6 enterprise domains:
1. Healthcare & Clinical Consultation (Doctor-Patient Diarization, Med-WER)
2. Financial Earnings Call (Q4 revenue, EPS, ticker symbols, NumER)
3. Legal Courtroom Proceedings (Judge, Defense, Prosecutor, legal citations)
4. AI & Software Engineering (Transformers, attention mechanisms, latency)
5. Customer Support / Contact Center (Telephony 8kHz, background chatter)
6. Multilingual Speech & Translation (Spanish, French, Hindi to English)
Includes synthetic modulated speech waveform generator.
"""

import os
import wave
import struct
import math
from typing import Dict, Any, List

SAMPLE_DATASETS = {
    "🏥 Medical & Healthcare Consultation": {
        "domain": "Healthcare",
        "description": "Doctor-Patient clinical dialogue covering symptoms, acute hypertension, and prescription dosages.",
        "ground_truth": (
            "The patient presented with acute hypertension and mild tachycardia. We recommended administering "
            "twenty milligrams of Lisinopril once daily along with continuous arterial blood pressure monitoring "
            "and Metformin five hundred milligrams twice a day."
        ),
        "speakers": [
            {"speaker": "Doctor", "text": "The patient presented with acute hypertension and mild tachycardia. We recommended administering twenty milligrams of Lisinopril once daily.", "start": 0.0, "end": 6.8},
            {"speaker": "Patient", "text": "Along with continuous arterial blood pressure monitoring and Metformin five hundred milligrams twice a day.", "start": 6.8, "end": 13.5}
        ],
        "entities": ["hypertension", "tachycardia", "Lisinopril", "twenty milligrams", "arterial", "Metformin", "five hundred milligrams"],
        "audio_duration": 13.5,
        "language": "en"
    },
    "📈 Financial Earnings & Banking Call": {
        "domain": "Finance",
        "description": "Quarterly earnings conference call with YoY growth, revenue in billions, operating margins, and EPS figures.",
        "ground_truth": (
            "In the fourth quarter, total revenue reached fourteen point two billion dollars, reflecting an eighteen percent "
            "year over year growth. Operating margins expanded to thirty two percent and diluted EPS was three dollars and forty cents."
        ),
        "speakers": [
            {"speaker": "CFO", "text": "In the fourth quarter, total revenue reached fourteen point two billion dollars, reflecting an eighteen percent year over year growth.", "start": 0.0, "end": 6.2},
            {"speaker": "CEO", "text": "Operating margins expanded to thirty two percent and diluted EPS was three dollars and forty cents.", "start": 6.2, "end": 11.8}
        ],
        "entities": ["fourth quarter", "fourteen point two billion dollars", "eighteen percent", "thirty two percent", "EPS", "three dollars and forty cents"],
        "audio_duration": 11.8,
        "language": "en"
    },
    "⚖️ Legal & Courtroom Proceedings": {
        "domain": "Legal",
        "description": "Courtroom litigation dialogue between Judge, Defense Counsel, and Prosecutor with statutory references.",
        "ground_truth": (
            "Your Honor, the defense objects under Section thirty two subsection B of the Federal Rules of Evidence. "
            "The prosecution has failed to establish proper chain of custody for exhibit number four."
        ),
        "speakers": [
            {"speaker": "Defense Counsel", "text": "Your Honor, the defense objects under Section thirty two subsection B of the Federal Rules of Evidence.", "start": 0.0, "end": 5.4},
            {"speaker": "Judge", "text": "The prosecution has failed to establish proper chain of custody for exhibit number four.", "start": 5.4, "end": 10.5}
        ],
        "entities": ["Section thirty two subsection B", "Federal Rules of Evidence", "prosecution", "chain of custody", "exhibit number four"],
        "audio_duration": 10.5,
        "language": "en"
    },
    "🤖 AI & Software Engineering Talk": {
        "domain": "Technology",
        "description": "Technical conference keynote explaining transformer self-attention, CTranslate2, and GPU inference latency.",
        "ground_truth": (
            "The transformer architecture relies entirely on self-attention mechanisms to compute representations "
            "of its input and output without using sequence-aligned recurrent neural networks or convolution. "
            "Faster Whisper achieves significant speedups by using CTranslate2 for efficient inference on CPU and GPU."
        ),
        "speakers": [
            {"speaker": "Speaker", "text": "The transformer architecture relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned recurrent neural networks or convolution. Faster Whisper achieves significant speedups by using CTranslate2 for efficient inference on CPU and GPU.", "start": 0.0, "end": 14.5}
        ],
        "entities": ["transformer", "self-attention", "recurrent neural networks", "convolution", "Faster Whisper", "CTranslate2", "CPU", "GPU"],
        "audio_duration": 14.5,
        "language": "en"
    },
    "📞 Customer Support Telephony (8kHz)": {
        "domain": "Contact Center",
        "description": "Customer service phone call with background chatter, order tracking numbers, and account verification.",
        "ground_truth": (
            "Thank you for calling customer support. My order number is nine eight four seven two, and I am calling "
            "to check whether my shipment has been processed and if there is any update regarding the express delivery."
        ),
        "speakers": [
            {"speaker": "Agent", "text": "Thank you for calling customer support.", "start": 0.0, "end": 2.2},
            {"speaker": "Customer", "text": "My order number is nine eight four seven two, and I am calling to check whether my shipment has been processed and if there is any update regarding the express delivery.", "start": 2.2, "end": 11.0}
        ],
        "entities": ["nine eight four seven two", "shipment", "express delivery"],
        "audio_duration": 11.0,
        "language": "en"
    },
    "🌐 Multilingual Speech & Translation (Spanish -> English)": {
        "domain": "Multilingual Translation",
        "description": "Spoken Spanish audio with ground-truth English translation for cross-lingual speech translation benchmarking.",
        "ground_truth": (
            "Bienvenidos a la conferencia internacional sobre inteligencia artificial y procesamiento de lenguaje natural en tiempo real."
        ),
        "reference_translation": (
            "Welcome to the international conference on artificial intelligence and natural language processing in real time."
        ),
        "speakers": [
            {"speaker": "Speaker", "text": "Bienvenidos a la conferencia internacional sobre inteligencia artificial y procesamiento de lenguaje natural en tiempo real.", "start": 0.0, "end": 8.5}
        ],
        "entities": ["inteligencia artificial", "lenguaje natural", "tiempo real"],
        "audio_duration": 8.5,
        "language": "es"
    }
}


def generate_sample_audio_wav(output_path: str, duration_sec: float = 3.0, sample_rate: int = 16000) -> str:
    """
    Generates a synthetic speech-frequency modulated sine wave audio file
    so users can immediately test the entire pipeline without uploading files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            # Harmonic vocal formant sweep
            freq = 220 + 80 * math.sin(2 * math.pi * 1.5 * t)
            amplitude = 16000 * (0.8 + 0.2 * math.sin(2 * math.pi * 0.5 * t))
            value = int(amplitude * math.sin(2 * math.pi * freq * t))
            frames.extend(struct.pack('<h', value))
            
        wav_file.writeframes(frames)
    return output_path
