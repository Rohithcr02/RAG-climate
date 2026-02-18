import os

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati", "bn": "Bengali",
    "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
}


def load_model() -> WhisperModel:
    """Load Whisper model. Size set via WHISPER_MODEL env var, defaults to small."""
    model_size = os.environ.get("WHISPER_MODEL", "small")
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def decode_audio(audio_file) -> np.ndarray:
    """Read WAV bytes into a mono float32 array at 16kHz."""
    audio_file.seek(0)
    data, samplerate = sf.read(audio_file, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if samplerate != 16000:
        import scipy.signal
        target_len = int(len(data) * 16000 / samplerate)
        data = scipy.signal.resample(data, target_len).astype(np.float32)
    return data


def transcribe(model: WhisperModel, audio_np: np.ndarray) -> tuple:
    """Transcribe audio and translate to English. Returns (text, status)."""
    segments, info = model.transcribe(
        audio_np,
        task="translate",
        beam_size=5,
        vad_filter=True,
        language=None,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    detected = info.language
    if detected == "en":
        status = "Transcribed from English"
    else:
        lang = LANG_NAMES.get(detected, detected.upper())
        status = f"Transcribed from {lang}, translated to English"
    return text, status
