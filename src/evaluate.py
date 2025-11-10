"""
evaluate.py

Evaluasi model CNN SIMPLIcity pada test set.

Fitur:
1. Load best model (berdasarkan val_accuracy) dari outputs/models.
2. Hitung:
   - test_loss
   - test_accuracy
   - error_ratio = 1 - test_accuracy
3. Bangun dan simpan confusion matrix 10x10 sebagai PNG.
4. Visualisasi beberapa contoh prediksi:
   - gambar
   - true label
   - predicted label
   - top-3 probabilitas softmax
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from . import config
from . import dataset
from . import utils


def _get_class_names() -> List[str]:
    """
    Karena dataset flat, kita pakai nama kelas generik:
    class_0, class_1, ..., class_9.
    """
    return [f"class_{i}" for i in range(config.NUM_CLASSES)]


def load_best_model() -> tf.keras.Model:
    """
    Load model terbaik dari BEST_MODEL_PATH.
    """
    model_path = config.BEST_MODEL_PATH
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Best model tidak ditemukan di: {model_path}\n"
            "Pastikan train.py sudah dijalankan dan ModelCheckpoint aktif."
        )

    print(f"==> Loading best model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    return model


def evaluate_on_test(model: tf.keras.Model) -> Tuple[float, float, float]:
    """
    Evaluasi model pada test set.

    Return
    ------
    (test_loss, test_accuracy, error_ratio)
    """
    print("==> Loading test dataset...")
    test_ds = dataset.get_test_dataset(batch_size=config.BATCH_SIZE)

    print("==> Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    error_ratio = 1.0 - float(test_acc)

    print(f"Test loss      : {test_loss:.4f}")
    print(f"Test accuracy  : {test_acc:.4f}")
    print(f"Error ratio    : {error_ratio:.4f}")

    return float(test_loss), float(test_acc), error_ratio


def build_confusion_matrix(
    model: tf.keras.Model,
    save_path: Path = None,
) -> np.ndarray:
    """
    Menghitung confusion matrix pada test set dan menyimpannya sebagai gambar PNG.

    Parameter
    ---------
    model : tf.keras.Model
        Model terlatih.
    save_path : Path
        Lokasi file PNG untuk confusion matrix. Jika None, gunakan config.CONF_MAT_FIG_PATH.

    Return
    ------
    np.ndarray
        Confusion matrix (NUM_CLASSES x NUM_CLASSES).
    """
    if save_path is None:
        save_path = config.CONF_MAT_FIG_PATH
    save_path = Path(save_path)

    print("==> Building confusion matrix on test set...")
    test_ds = dataset.get_test_dataset(batch_size=config.BATCH_SIZE)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        # labels: one-hot -> class index
        true_classes = tf.argmax(labels, axis=1).numpy()

        # prediksi
        probs = model.predict(images, verbose=0)
        pred_classes = np.argmax(probs, axis=1)

        y_true.extend(true_classes.tolist())
        y_pred.extend(pred_classes.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    class_names = _get_class_names()

    # Plot confusion matrix
    utils.ensure_directories()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix disimpan di: {save_path}")
    return cm


def visualize_sample_predictions(
    model: tf.keras.Model,
    num_examples: int = 9,
    save_path: Path = None,
) -> None:
    """
    Visualisasi beberapa contoh prediksi dari test set.

    - Menampilkan grid gambar (misal 3x3)
    - Title tiap gambar:
        T: true_class / P: pred_class
        top-3: class (prob%)

    Parameter
    ---------
    model : tf.keras.Model
        Model terlatih.
    num_examples : int
        Jumlah contoh yang divisualisasikan (maks 16 disarankan).
    save_path : Path
        Lokasi file PNG. Jika None, simpan sebagai 'sample_predictions_test.png'
        di folder figures.
    """
    if save_path is None:
        save_path = Path(config.FIGURE_DIR) / "sample_predictions_test.png"
    save_path = Path(save_path)

    print(f"==> Visualizing {num_examples} sample predictions...")
    test_ds = dataset.get_test_dataset(batch_size=config.BATCH_SIZE)

    # unbatch supaya bisa ambil N contoh
    test_unbatched = test_ds.unbatch().take(num_examples)

    images_list = []
    labels_list = []
    for img, label in test_unbatched:
        images_list.append(img.numpy())
        labels_list.append(label.numpy())

    if not images_list:
        print("Tidak ada data di test set untuk divisualisasikan.")
        return

    images_arr = np.stack(images_list, axis=0)
    labels_arr = np.stack(labels_list, axis=0)

    # prediksi
    probs = model.predict(images_arr, verbose=0)
    true_classes = np.argmax(labels_arr, axis=1)
    pred_classes = np.argmax(probs, axis=1)

    class_names = _get_class_names()

    # grid: sebisa mungkin square (misal 3x3, 4x4)
    n = len(images_arr)
    grid_size = int(np.ceil(np.sqrt(n)))

    utils.ensure_directories()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3 * grid_size, 3 * grid_size))

    for i in range(n):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        img = images_arr[i]

        # img disimpan sdh normalized 0..1
        ax.imshow(img)
        ax.axis("off")

        true_id = int(true_classes[i])
        pred_id = int(pred_classes[i])

        # top-3 probabilitas
        top3_idx = np.argsort(probs[i])[-3:][::-1]
        top3_info = ", ".join(
            [f"{class_names[j]} ({probs[i][j]*100:.1f}%)" for j in top3_idx]
        )

        title = f"T: {class_names[true_id]}\nP: {class_names[pred_id]}\n{top3_info}"
        ax.set_title(title, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Contoh prediksi disimpan di: {save_path}")


def main() -> None:
    """
    Pipeline evaluasi lengkap:
    1. Set seed & pastikan direktori output ada.
    2. Load best model.
    3. Evaluasi test set (loss, accuracy, error ratio).
    4. Bangun confusion matrix dan simpan.
    5. Visualisasi beberapa sample prediksi.
    """
    utils.set_seed()
    utils.ensure_directories()

    model = load_best_model()

    print("==> Evaluating model on test set...")
    test_loss, test_acc, error_ratio = evaluate_on_test(model)

    print("\n==> Building confusion matrix...")
    cm = build_confusion_matrix(model)

    print("\n==> Visualizing sample predictions...")
    visualize_sample_predictions(model, num_examples=9)

    print("\nRingkasan:")
    print(f"- Test loss     : {test_loss:.4f}")
    print(f"- Test accuracy : {test_acc:.4f}")
    print(f"- Error ratio   : {error_ratio:.4f}")
    print(f"- Confusion matrix shape: {cm.shape}")


if __name__ == "__main__":
    # Jalankan dari root project:
    # $ python src/evaluate.py
    # atau
    # $ python -m src.evaluate
    main()
