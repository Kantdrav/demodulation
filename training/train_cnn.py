import argparse
import json
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MFCC CNN on folder-based audio dataset")
    parser.add_argument("--dataset-dir", required=True, help="Root dataset directory")
    parser.add_argument("--output-model", required=True, help="Path to output .h5 model")
    parser.add_argument("--output-config", required=True, help="Path to output model_config.json")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=173)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=8000.0,
        help="Low-pass cutoff frequency used before MFCC extraction",
    )
    parser.add_argument(
        "--disable-lowpass",
        action="store_true",
        help="Disable low-pass preprocessing (enabled by default)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="training/artifacts",
        help="Directory to save plots and evaluation artifacts",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=0,
        help="Optional cap for quick experiments; 0 means all files",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="Number of augmented copies to generate per source file (0 disables augmentation)",
    )
    parser.add_argument(
        "--augment-noise-min-snr-db",
        type=float,
        default=8.0,
        help="Minimum SNR in dB for additive Gaussian noise augmentation",
    )
    parser.add_argument(
        "--augment-noise-max-snr-db",
        type=float,
        default=20.0,
        help="Maximum SNR in dB for additive Gaussian noise augmentation",
    )
    parser.add_argument(
        "--augment-gain-min-db",
        type=float,
        default=-4.0,
        help="Minimum random gain in dB for augmentation",
    )
    parser.add_argument(
        "--augment-gain-max-db",
        type=float,
        default=4.0,
        help="Maximum random gain in dB for augmentation",
    )
    parser.add_argument(
        "--augment-shift-max-fraction",
        type=float,
        default=0.1,
        help="Maximum circular time shift as fraction of waveform length",
    )
    parser.add_argument(
        "--noisy-eval-copies",
        type=int,
        default=1,
        help="Number of noisy copies per validation/test source file for robustness evaluation (0 disables)",
    )
    parser.add_argument(
        "--noisy-eval-noise-min-snr-db",
        type=float,
        default=5.0,
        help="Minimum SNR in dB for noisy validation/test evaluation",
    )
    parser.add_argument(
        "--noisy-eval-noise-max-snr-db",
        type=float,
        default=15.0,
        help="Maximum SNR in dB for noisy validation/test evaluation",
    )
    parser.add_argument(
        "--noisy-eval-gain-min-db",
        type=float,
        default=-3.0,
        help="Minimum random gain in dB for noisy validation/test evaluation",
    )
    parser.add_argument(
        "--noisy-eval-gain-max-db",
        type=float,
        default=3.0,
        help="Maximum random gain in dB for noisy validation/test evaluation",
    )
    parser.add_argument(
        "--noisy-eval-shift-max-fraction",
        type=float,
        default=0.05,
        help="Maximum circular time shift fraction for noisy validation/test evaluation",
    )
    return parser.parse_args()


def list_class_dirs(dataset_dir: Path) -> list[Path]:
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found inside: {dataset_dir}")
    return class_dirs


def collect_audio_files(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS])


def pad_or_truncate(mfcc: np.ndarray, max_len: int) -> np.ndarray:
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def lowpass_filter_audio(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
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


def extract_mfcc(
    path: Path,
    sample_rate: int,
    n_mfcc: int,
    max_len: int,
    apply_lowpass: bool,
    lowpass_cutoff_hz: float,
) -> np.ndarray:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("Invalid or empty audio")
    return extract_mfcc_from_waveform(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        max_len=max_len,
        apply_lowpass=apply_lowpass,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )


def extract_mfcc_from_waveform(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    max_len: int,
    apply_lowpass: bool,
    lowpass_cutoff_hz: float,
) -> np.ndarray:
    if apply_lowpass:
        y = lowpass_filter_audio(y, sr, lowpass_cutoff_hz)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return pad_or_truncate(mfcc, max_len)


def add_noise_at_snr(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.square(y)))
    if signal_power <= 0:
        return np.asarray(y, dtype=np.float32)

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return np.asarray(y + noise, dtype=np.float32)


def apply_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20.0)
    return np.asarray(y * gain, dtype=np.float32)


def apply_time_shift(y: np.ndarray, max_shift_fraction: float, rng: np.random.Generator) -> np.ndarray:
    if y.size == 0 or max_shift_fraction <= 0:
        return np.asarray(y, dtype=np.float32)
    max_shift = int(max(1, round(y.size * max_shift_fraction)))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.asarray(np.roll(y, shift), dtype=np.float32)


def augment_waveform(
    y: np.ndarray,
    rng: np.random.Generator,
    noise_min_snr_db: float,
    noise_max_snr_db: float,
    gain_min_db: float,
    gain_max_db: float,
    shift_max_fraction: float,
) -> np.ndarray:
    augmented = np.asarray(y, dtype=np.float32)

    snr_db = float(rng.uniform(noise_min_snr_db, noise_max_snr_db))
    augmented = add_noise_at_snr(augmented, snr_db=snr_db, rng=rng)

    gain_db = float(rng.uniform(gain_min_db, gain_max_db))
    augmented = apply_gain(augmented, gain_db=gain_db)

    augmented = apply_time_shift(augmented, max_shift_fraction=shift_max_fraction, rng=rng)

    peak = np.max(np.abs(augmented)) if augmented.size else 0.0
    if peak > 1.0:
        augmented = augmented / peak

    return np.asarray(augmented, dtype=np.float32)


def build_model(n_mfcc: int, max_len: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_mfcc, max_len, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_val_test_split(
    indices: list[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rnd = random.Random(seed)
    shuffled = indices[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    if n <= 1:
        return shuffled, [], []

    val_count = int(round(n * val_ratio)) if val_ratio > 0 else 0
    test_count = int(round(n * test_ratio)) if test_ratio > 0 else 0

    if test_ratio > 0 and n >= 3 and test_count == 0:
        test_count = 1
    if val_ratio > 0 and n - test_count >= 2 and val_count == 0:
        val_count = 1

    max_non_train = n - 1
    if val_count + test_count > max_non_train:
        overflow = val_count + test_count - max_non_train
        reduce_val = min(val_count, overflow)
        val_count -= reduce_val
        overflow -= reduce_val
        test_count = max(0, test_count - overflow)

    train_end = n - (val_count + test_count)
    val_end = n - test_count
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_training_curves(history: tf.keras.callbacks.History, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "validation_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "loss_curve.png", dpi=150)
    plt.close()


def save_confusion_matrix_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    artifacts_dir: Path,
    split_name: str,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = split_name.lower().replace(" ", "_")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.savetxt(artifacts_dir / f"confusion_matrix_{safe_name}.csv", cm, delimiter=",", fmt="%d")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (artifacts_dir / f"classification_report_{safe_name}.txt").write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({split_name.title()} Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(artifacts_dir / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.close(fig)


def evaluate_split(
    model: tf.keras.Model,
    x_data: np.ndarray,
    y_data: np.ndarray,
    class_names: list[str],
    artifacts_dir: Path,
    split_name: str,
) -> dict:
    if len(y_data) == 0:
        return {"samples": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None}

    y_pred_prob = model.predict(x_data, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    save_confusion_matrix_artifacts(y_data, y_pred, class_names, artifacts_dir, split_name)

    report_dict = classification_report(
        y_data,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": int(len(y_data)),
        "accuracy": float(np.mean(y_pred == y_data)),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }


def build_feature_set(
    records: list[tuple[Path, int]],
    sample_rate: int,
    n_mfcc: int,
    max_len: int,
    apply_lowpass: bool,
    lowpass_cutoff_hz: float,
    augment_copies: int,
    rng: np.random.Generator,
    noise_min_snr_db: float,
    noise_max_snr_db: float,
    gain_min_db: float,
    gain_max_db: float,
    shift_max_fraction: float,
    include_clean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for audio_path, class_idx in records:
        try:
            y_audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            if y_audio.size == 0:
                raise ValueError("Invalid or empty audio")

            if include_clean:
                mfcc = extract_mfcc_from_waveform(
                    y=y_audio,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    max_len=max_len,
                    apply_lowpass=apply_lowpass,
                    lowpass_cutoff_hz=lowpass_cutoff_hz,
                )
                features.append(mfcc)
                labels.append(class_idx)

            for _ in range(augment_copies):
                y_aug = augment_waveform(
                    y_audio,
                    rng=rng,
                    noise_min_snr_db=noise_min_snr_db,
                    noise_max_snr_db=noise_max_snr_db,
                    gain_min_db=gain_min_db,
                    gain_max_db=gain_max_db,
                    shift_max_fraction=shift_max_fraction,
                )
                mfcc_aug = extract_mfcc_from_waveform(
                    y=y_aug,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    max_len=max_len,
                    apply_lowpass=apply_lowpass,
                    lowpass_cutoff_hz=lowpass_cutoff_hz,
                )
                features.append(mfcc_aug)
                labels.append(class_idx)
        except Exception as exc:
            print(f"Skipping {audio_path}: {exc}")

    if not features:
        x_out = np.empty((0, n_mfcc, max_len, 1), dtype=np.float32)
        y_out = np.empty((0,), dtype=np.int32)
        return x_out, y_out

    x_out = np.asarray(features, dtype=np.float32)
    x_out = np.expand_dims(x_out, axis=-1)
    y_out = np.asarray(labels, dtype=np.int32)
    return x_out, y_out


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be < 1")
    if args.augment_copies < 0:
        raise ValueError("--augment-copies must be >= 0")
    if args.augment_noise_min_snr_db > args.augment_noise_max_snr_db:
        raise ValueError("--augment-noise-min-snr-db must be <= --augment-noise-max-snr-db")
    if args.augment_gain_min_db > args.augment_gain_max_db:
        raise ValueError("--augment-gain-min-db must be <= --augment-gain-max-db")
    if not 0 <= args.augment_shift_max_fraction <= 1:
        raise ValueError("--augment-shift-max-fraction must be in [0, 1]")
    if args.noisy_eval_copies < 0:
        raise ValueError("--noisy-eval-copies must be >= 0")
    if args.noisy_eval_noise_min_snr_db > args.noisy_eval_noise_max_snr_db:
        raise ValueError("--noisy-eval-noise-min-snr-db must be <= --noisy-eval-noise-max-snr-db")
    if args.noisy_eval_gain_min_db > args.noisy_eval_gain_max_db:
        raise ValueError("--noisy-eval-gain-min-db must be <= --noisy-eval-gain-max-db")
    if not 0 <= args.noisy_eval_shift_max_fraction <= 1:
        raise ValueError("--noisy-eval-shift-max-fraction must be in [0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")
    rng = np.random.default_rng(args.seed)

    class_dirs = list_class_dirs(dataset_dir)
    class_names = [p.name for p in class_dirs]
    print(f"Classes: {class_names}")

    records: list[tuple[Path, int]] = []

    for class_idx, class_dir in enumerate(class_dirs):
        files = collect_audio_files(class_dir)
        if args.max_files_per_class > 0:
            files = files[: args.max_files_per_class]

        if not files:
            raise ValueError(f"No audio files found under class folder: {class_dir}")

        print(f"Found {len(files)} files in {class_dir.name}")
        for audio_path in files:
            records.append((audio_path, class_idx))

    if not records:
        raise ValueError("No usable audio files found after dataset scan")

    labels_array = np.array([class_idx for _, class_idx in records], dtype=np.int32)

    train_idx = []
    val_idx = []
    test_idx = []
    for class_idx in range(len(class_names)):
        cls_indices = np.where(labels_array == class_idx)[0].tolist()
        tr, va, te = train_val_test_split(
            cls_indices,
            args.val_ratio,
            args.test_ratio,
            args.seed + class_idx,
        )
        train_idx.extend(tr)
        val_idx.extend(va)
        test_idx.extend(te)

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    x_train, y_train = build_feature_set(
        records=train_records,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        max_len=args.max_len,
        apply_lowpass=not args.disable_lowpass,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        augment_copies=args.augment_copies,
        rng=rng,
        noise_min_snr_db=args.augment_noise_min_snr_db,
        noise_max_snr_db=args.augment_noise_max_snr_db,
        gain_min_db=args.augment_gain_min_db,
        gain_max_db=args.augment_gain_max_db,
        shift_max_fraction=args.augment_shift_max_fraction,
        include_clean=True,
    )
    x_val, y_val = build_feature_set(
        records=val_records,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        max_len=args.max_len,
        apply_lowpass=not args.disable_lowpass,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        augment_copies=0,
        rng=rng,
        noise_min_snr_db=args.augment_noise_min_snr_db,
        noise_max_snr_db=args.augment_noise_max_snr_db,
        gain_min_db=args.augment_gain_min_db,
        gain_max_db=args.augment_gain_max_db,
        shift_max_fraction=args.augment_shift_max_fraction,
        include_clean=True,
    )
    x_test, y_test = build_feature_set(
        records=test_records,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        max_len=args.max_len,
        apply_lowpass=not args.disable_lowpass,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        augment_copies=0,
        rng=rng,
        noise_min_snr_db=args.augment_noise_min_snr_db,
        noise_max_snr_db=args.augment_noise_max_snr_db,
        gain_min_db=args.augment_gain_min_db,
        gain_max_db=args.augment_gain_max_db,
        shift_max_fraction=args.augment_shift_max_fraction,
        include_clean=True,
    )

    print(f"Train source files: {len(train_records)}")
    print(f"Validation source files: {len(val_records)}")
    print(f"Test source files: {len(test_records)}")
    print(f"Train feature samples (after augmentation): {len(x_train)}")
    print(f"Validation feature samples: {len(x_val)}")
    print(f"Test feature samples: {len(x_test)}")

    if len(x_val) == 0:
        raise ValueError("Validation split is empty. Increase dataset size or --val-ratio.")

    model = build_model(args.n_mfcc, args.max_len, len(class_names))

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=artifacts_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)

    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "sample_rate": args.sample_rate,
        "n_mfcc": args.n_mfcc,
        "max_len": args.max_len,
        "lowpass_cutoff_hz": None if args.disable_lowpass else args.lowpass_cutoff_hz,
        "class_names": class_names,
    }
    output_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_training_curves(history, artifacts_dir)

    val_metrics = evaluate_split(model, x_val, y_val, class_names, artifacts_dir, "validation")
    test_metrics = evaluate_split(model, x_test, y_test, class_names, artifacts_dir, "test")

    noisy_val_metrics = None
    noisy_test_metrics = None
    if args.noisy_eval_copies > 0:
        x_val_noisy, y_val_noisy = build_feature_set(
            records=val_records,
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            max_len=args.max_len,
            apply_lowpass=not args.disable_lowpass,
            lowpass_cutoff_hz=args.lowpass_cutoff_hz,
            augment_copies=args.noisy_eval_copies,
            rng=rng,
            noise_min_snr_db=args.noisy_eval_noise_min_snr_db,
            noise_max_snr_db=args.noisy_eval_noise_max_snr_db,
            gain_min_db=args.noisy_eval_gain_min_db,
            gain_max_db=args.noisy_eval_gain_max_db,
            shift_max_fraction=args.noisy_eval_shift_max_fraction,
            include_clean=False,
        )
        x_test_noisy, y_test_noisy = build_feature_set(
            records=test_records,
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            max_len=args.max_len,
            apply_lowpass=not args.disable_lowpass,
            lowpass_cutoff_hz=args.lowpass_cutoff_hz,
            augment_copies=args.noisy_eval_copies,
            rng=rng,
            noise_min_snr_db=args.noisy_eval_noise_min_snr_db,
            noise_max_snr_db=args.noisy_eval_noise_max_snr_db,
            gain_min_db=args.noisy_eval_gain_min_db,
            gain_max_db=args.noisy_eval_gain_max_db,
            shift_max_fraction=args.noisy_eval_shift_max_fraction,
            include_clean=False,
        )
        noisy_val_metrics = evaluate_split(
            model,
            x_val_noisy,
            y_val_noisy,
            class_names,
            artifacts_dir,
            "validation_noisy",
        )
        noisy_test_metrics = evaluate_split(
            model,
            x_test_noisy,
            y_test_noisy,
            class_names,
            artifacts_dir,
            "test_noisy",
        )

    metrics_summary = {
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "n_mfcc": args.n_mfcc,
        "max_len": args.max_len,
        "lowpass_enabled": not args.disable_lowpass,
        "lowpass_cutoff_hz": args.lowpass_cutoff_hz,
        "augment_copies": args.augment_copies,
        "augment_noise_min_snr_db": args.augment_noise_min_snr_db,
        "augment_noise_max_snr_db": args.augment_noise_max_snr_db,
        "augment_gain_min_db": args.augment_gain_min_db,
        "augment_gain_max_db": args.augment_gain_max_db,
        "augment_shift_max_fraction": args.augment_shift_max_fraction,
        "noisy_eval_copies": args.noisy_eval_copies,
        "noisy_eval_noise_min_snr_db": args.noisy_eval_noise_min_snr_db,
        "noisy_eval_noise_max_snr_db": args.noisy_eval_noise_max_snr_db,
        "noisy_eval_gain_min_db": args.noisy_eval_gain_min_db,
        "noisy_eval_gain_max_db": args.noisy_eval_gain_max_db,
        "noisy_eval_shift_max_fraction": args.noisy_eval_shift_max_fraction,
        "validation": val_metrics,
        "test": test_metrics,
        "validation_noisy": noisy_val_metrics,
        "test_noisy": noisy_test_metrics,
    }
    (artifacts_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model to: {output_model}")
    print(f"Saved config to: {output_config}")
    print(f"Saved training artifacts to: {artifacts_dir}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Validation noisy metrics: {noisy_val_metrics}")
    print(f"Test noisy metrics: {noisy_test_metrics}")


if __name__ == "__main__":
    main()
