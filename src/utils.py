"""
utils.py

Fungsi bantu umum untuk proyek:
- set random seed
- memastikan direktori output terbentuk
- menyimpan history training
- membuat plot loss & accuracy
"""

import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import config


def set_seed(seed: int = None) -> None:
    """
    Set random seed untuk Python, NumPy, dan TensorFlow.

    Parameter
    ---------
    seed : int
        Nilai seed. Jika None, gunakan config.RANDOM_SEED.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories() -> None:
    """
    Membuat direktori output yang diperlukan jika belum ada.
    """
    for path in [
        config.LOG_DIR,
        config.MODEL_DIR,
        config.FIGURE_DIR,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def history_to_dict(history: Any) -> Dict[str, Any]:
    """
    Konversi objek History Keras menjadi dict biasa.

    Parameter
    ---------
    history : keras.callbacks.History atau dict

    Return
    ------
    dict
        Dictionary metric -> list nilai per epoch.
    """
    if history is None:
        return {}

    if isinstance(history, dict):
        return history

    # Keras History: history.history berisi dict
    if hasattr(history, "history"):
        return dict(history.history)

    raise ValueError("Tipe history tidak dikenali. Harus dict atau keras.callbacks.History.")


def save_history(history: Any,
                 csv_path: Path = None,
                 json_path: Path = None) -> None:
    """
    Simpan history training ke CSV dan JSON.

    Parameter
    ---------
    history : History atau dict
        History yang dikembalikan oleh model.fit().
    csv_path : Path
        Lokasi file CSV.
    json_path : Path
        Lokasi file JSON.
    """
    hist_dict = history_to_dict(history)

    if csv_path is None:
        csv_path = config.HISTORY_CSV_PATH
    if json_path is None:
        json_path = config.HISTORY_JSON_PATH

    csv_path = Path(csv_path)
    json_path = Path(json_path)

    # Simpan JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    # Simpan CSV manual (tanpa pandas, biar ringan)
    keys = list(hist_dict.keys())
    num_epochs = len(next(iter(hist_dict.values()), []))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        # header
        f.write(",".join(keys) + "\n")
        # baris per epoch
        for i in range(num_epochs):
            row = []
            for k in keys:
                values = hist_dict.get(k, [])
                value = values[i] if i < len(values) else ""
                row.append(str(value))
            f.write(",".join(row) + "\n")


def plot_training_curves(history: Any,
                         loss_path: Path = None,
                         acc_path: Path = None) -> None:
    """
    Plot kurva loss dan accuracy (train & validation) ke file PNG.

    Parameter
    ---------
    history : History atau dict
        History dari model.fit().
    loss_path : Path
        Lokasi file gambar loss.
    acc_path : Path
        Lokasi file gambar accuracy.
    """
    hist = history_to_dict(history)

    if loss_path is None:
        loss_path = config.LOSS_FIG_PATH
    if acc_path is None:
        acc_path = config.ACC_FIG_PATH

    loss_path = Path(loss_path)
    acc_path = Path(acc_path)

    # Pastikan direktori ada
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    acc_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------- Plot Loss --------------------
    plt.figure()
    train_loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    plt.plot(train_loss, label="Train Loss")
    if val_loss:
        plt.plot(val_loss, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # -------------------- Plot Accuracy --------------------
    plt.figure()
    train_acc = hist.get("accuracy", hist.get("acc", []))
    val_acc = hist.get("val_accuracy", hist.get("val_acc", []))

    plt.plot(train_acc, label="Train Accuracy")
    if val_acc:
        plt.plot(val_acc, label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()


def get_class_names_from_directory(train_dir: Path = None):
    """
    Mengambil daftar nama kelas dari subfolder di direktori train.

    Parameter
    ---------
    train_dir : Path
        Path ke direktori train/ yang berisi subfolder per kelas.

    Return
    ------
    List[str]
        Nama-nama kelas tersortir alfabetis.
    """
    if train_dir is None:
        train_dir = config.TRAIN_DIR

    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory tidak ditemukan: {train_dir}")

    class_names = sorted(
        [p.name for p in train_dir.iterdir() if p.is_dir()]
    )
    return class_names
