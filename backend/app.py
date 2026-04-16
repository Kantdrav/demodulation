import io
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import librosa
import numpy as np
import requests
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

AUPHONIC_API_KEY = os.getenv("AUPHONIC_API_KEY", "").strip()
AUPHONIC_PRESET = os.getenv("AUPHONIC_PRESET", "").strip()
AUPHONIC_TIMEOUT_S = int(os.getenv("AUPHONIC_TIMEOUT_S", "300"))
AUPHONIC_POLL_INTERVAL_S = float(os.getenv("AUPHONIC_POLL_INTERVAL_S", "2.0"))
AUPHONIC_BASE_URL = "https://auphonic.com/api"


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
LOWPASS_CUTOFF_HZ = float(os.getenv("LOWPASS_CUTOFF_HZ", "8000"))

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


def lowpass_filter_audio(y: np.ndarray, sr: int, cutoff_hz: float = LOWPASS_CUTOFF_HZ) -> np.ndarray:
    if y.size == 0:
        return y

    nyquist = sr / 2.0
    effective_cutoff = max(1.0, min(float(cutoff_hz), nyquist * 0.95))
    if effective_cutoff >= nyquist:
        return np.asarray(y, dtype=np.float32)

    spectrum = np.fft.rfft(y)
    frequencies = np.fft.rfftfreq(len(y), d=1.0 / sr)
    spectrum[frequencies > effective_cutoff] = 0
    filtered = np.fft.irfft(spectrum, n=len(y))
    return np.asarray(filtered, dtype=np.float32)


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

    y = lowpass_filter_audio(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = pad_or_truncate(mfcc, MAX_LEN)

    x = np.expand_dims(mfcc, axis=-1)
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x


def denoise_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> io.BytesIO:
    y, sr = decode_audio_bytes(audio_bytes, filename)
    if y.size == 0:
        raise ValueError("Empty or invalid audio")

    y = lowpass_filter_audio(y, sr)
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


def _build_auphonic_download_url(download_url: str, bearer_token: str) -> str:
    parsed = urlparse(download_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["bearer_token"] = bearer_token
    return urlunparse(parsed._replace(query=urlencode(query)))


def denoise_audio_bytes_auphonic(audio_bytes: bytes, filename: str | None = None) -> io.BytesIO:
    if not AUPHONIC_API_KEY:
        raise ValueError("AUPHONIC_API_KEY is not configured on server")
    if not AUPHONIC_PRESET:
        raise ValueError("AUPHONIC_PRESET is not configured on server")
    if not audio_bytes:
        raise ValueError("Empty or invalid audio")

    safe_name = Path(filename or "input.wav").name or "input.wav"
    headers = {"Authorization": f"bearer {AUPHONIC_API_KEY}"}

    create_response = requests.post(
        f"{AUPHONIC_BASE_URL}/simple/productions.json",
        headers=headers,
        data={
            "preset": AUPHONIC_PRESET,
            "title": f"Demod Denoise {int(time.time())}",
            "action": "start",
            # Keep denoise enabled even if preset settings differ.
            "denoise": "true",
        },
        files={"input_file": (safe_name, audio_bytes, "application/octet-stream")},
        timeout=60,
    )
    create_response.raise_for_status()
    create_payload = create_response.json()
    production_data = create_payload.get("data") or {}
    production_uuid = production_data.get("uuid")
    if not production_uuid:
        raise ValueError("Auphonic did not return a production UUID")

    deadline = time.time() + max(30, AUPHONIC_TIMEOUT_S)
    while time.time() < deadline:
        detail_response = requests.get(
            f"{AUPHONIC_BASE_URL}/production/{production_uuid}.json",
            headers=headers,
            timeout=30,
        )
        detail_response.raise_for_status()
        detail_payload = detail_response.json()
        detail_data = detail_payload.get("data") or {}
        status_string = str(detail_data.get("status_string", "")).lower()
        error_message = detail_data.get("error_message") or detail_payload.get("error_message")

        if "done" in status_string:
            output_files = detail_data.get("output_files") or []
            if not output_files:
                raise ValueError("Auphonic finished without output files")

            # Prefer WAV output if available, otherwise fallback to first file.
            output = next(
                (item for item in output_files if str(item.get("ending", "")).lower() == "wav"),
                output_files[0],
            )
            download_url = output.get("download_url")
            if not download_url:
                raise ValueError("Auphonic output file is missing download URL")

            download_response = requests.get(
                _build_auphonic_download_url(download_url, AUPHONIC_API_KEY),
                timeout=120,
                allow_redirects=True,
            )
            download_response.raise_for_status()
            out = io.BytesIO(download_response.content)
            out.seek(0)
            return out

        if error_message or any(word in status_string for word in ["error", "failed", "canceled"]):
            message = str(error_message or detail_data.get("warning_message") or "Auphonic failed")
            raise ValueError(message)

        time.sleep(max(0.5, AUPHONIC_POLL_INTERVAL_S))

    raise TimeoutError("Auphonic processing timed out")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "auphonic_configured": bool(AUPHONIC_API_KEY and AUPHONIC_PRESET),
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


@app.post("/denoise-auphonic")
async def denoise_auphonic(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        audio = await file.read()
        wav_buffer = denoise_audio_bytes_auphonic(audio, file.filename)
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=denoised_auphonic.wav"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Auphonic request failed: {str(exc)}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auphonic denoising failed: {str(exc)}") from exc
