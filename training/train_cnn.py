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
    parser.add_argument("--seed", type=int, default=42)
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


def extract_mfcc(path: Path, sample_rate: int, n_mfcc: int, max_len: int) -> np.ndarray:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("Invalid or empty audio")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return pad_or_truncate(mfcc, max_len)


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


def train_val_split(indices: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rnd = random.Random(seed)
    shuffled = indices[:]
    rnd.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - val_ratio)))
    split = min(split, len(shuffled) - 1)
    return shuffled[:split], shuffled[split:]


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
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.savetxt(artifacts_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")

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
        title="Confusion Matrix (Validation Set)",
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
    fig.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    class_dirs = list_class_dirs(dataset_dir)
    class_names = [p.name for p in class_dirs]
    print(f"Classes: {class_names}")

    features = []
    labels = []

    for class_idx, class_dir in enumerate(class_dirs):
        files = collect_audio_files(class_dir)
        if args.max_files_per_class > 0:
            files = files[: args.max_files_per_class]

        if not files:
            raise ValueError(f"No audio files found under class folder: {class_dir}")

        print(f"Loading {len(files)} files from {class_dir.name}")
        for audio_path in files:
            try:
                mfcc = extract_mfcc(audio_path, args.sample_rate, args.n_mfcc, args.max_len)
                features.append(mfcc)
                labels.append(class_idx)
            except Exception as exc:
                print(f"Skipping {audio_path}: {exc}")

    if not features:
        raise ValueError("No usable audio files found after preprocessing")

    x = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    x = np.expand_dims(x, axis=-1)
    print(f"Feature shape: {x.shape}")

    train_idx = []
    val_idx = []
    for class_idx in range(len(class_names)):
        cls_indices = np.where(y == class_idx)[0].tolist()
        tr, va = train_val_split(cls_indices, args.val_ratio, args.seed)
        train_idx.extend(tr)
        val_idx.extend(va)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")

    model = build_model(args.n_mfcc, args.max_len, len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
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
        "class_names": class_names,
    }
    output_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    y_pred_prob = model.predict(x_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    artifacts_dir = Path(args.artifacts_dir)
    save_training_curves(history, artifacts_dir)
    save_confusion_matrix_artifacts(y_val, y_pred, class_names, artifacts_dir)

    print(f"Saved model to: {output_model}")
    print(f"Saved config to: {output_config}")
    print(f"Saved training artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
