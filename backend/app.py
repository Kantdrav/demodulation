import io
import json
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "model_config.json")
DEFAULT_CONFIG = {
    "sample_rate": 22050,
    "n_mfcc": 40,
    "max_len": 173,
    "class_names": ["class_0", "class_1"],
}


def load_model_config(model_path: str, config_path: str) -> dict:
    config_candidate = Path(config_path)
    if not config_candidate.is_absolute():
        config_candidate = Path(model_path).parent / config_path

    if config_candidate.exists():
        with config_candidate.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {**DEFAULT_CONFIG, **data}

    return DEFAULT_CONFIG.copy()


MODEL_CONFIG = load_model_config(MODEL_PATH, CONFIG_PATH)
SAMPLE_RATE = int(MODEL_CONFIG["sample_rate"])
N_MFCC = int(MODEL_CONFIG["n_mfcc"])
MAX_LEN = int(MODEL_CONFIG["max_len"])

app = FastAPI(title="Audio Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as exc:
    model = None
    print(f"Model load error: {exc}")

CLASS_NAMES = list(MODEL_CONFIG.get("class_names", ["class_0", "class_1"]))


def pad_or_truncate(mfcc: np.ndarray, max_len: int) -> np.ndarray:
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def decode_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> tuple[np.ndarray, int]:
    if not audio_bytes:
        raise ValueError("Empty or invalid audio")

    # Try in-memory decoding first (fast path for wav/flac/ogg).
    try:
        return librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    except Exception:
        pass

    suffix = Path(filename or "").suffix.lower() or ".tmp"
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        return librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(
            "Unsupported or corrupted audio format. Upload WAV/MP3/M4A/FLAC/OGG."
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def preprocess_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> np.ndarray:
    y, sr = decode_audio_bytes(audio_bytes, filename)
    if y.size == 0:
        raise ValueError("Empty or invalid audio")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = pad_or_truncate(mfcc, MAX_LEN)

    x = np.expand_dims(mfcc, axis=-1)
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x


def denoise_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> io.BytesIO:
    y, sr = decode_audio_bytes(audio_bytes, filename)
    if y.size == 0:
        raise ValueError("Empty or invalid audio")

    n_fft = 1024
    hop_length = 256
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_frames = max(1, int((0.5 * sr) / hop_length))
    noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)

    alpha = 1.5
    beta = 0.05
    cleaned_mag = np.maximum(magnitude - alpha * noise_profile, beta * noise_profile)
    cleaned_stft = cleaned_mag * np.exp(1j * phase)

    y_denoised = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))
    y_denoised = np.asarray(y_denoised, dtype=np.float32)

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, y_denoised, sr, format="WAV")
    wav_buffer.seek(0)
    return wav_buffer


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "max_len": MAX_LEN,
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        audio = await file.read()
        x = preprocess_audio_bytes(audio, file.filename)

        pred = model.predict(x, verbose=0)
        pred = np.squeeze(pred)

        if np.ndim(pred) == 0:
            confidence = float(pred)
            class_idx = int(confidence >= 0.5)
        elif getattr(pred, "shape", ()) == (1,):
            confidence = float(pred[0])
            class_idx = int(confidence >= 0.5)
        else:
            class_idx = int(np.argmax(pred))
            confidence = float(pred[class_idx])

        label = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"

        return {
            "label": label,
            "class_index": class_idx,
            "confidence": confidence,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}") from exc


@app.post("/denoise")
async def denoise(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        audio = await file.read()
        wav_buffer = denoise_audio_bytes(audio, file.filename)
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=denoised.wav"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Denoising failed: {str(exc)}") from exc
