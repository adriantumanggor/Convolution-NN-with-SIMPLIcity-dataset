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
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from . import config
from . import dataset
from . import utils


# --- PERUBAHAN DI SINI ---
# Menggunakan nama kelas yang sesuai dengan dataset SIMPLIcity
def _get_class_names() -> List[str]:
    """
    Mengembalikan nama kelas aktual sesuai mapping dataset.
    Indeks 0 -> 'People', Indeks 1 -> 'Beach', ...
    """
    return [
        "People",     # 0-99
        "Beach",      # 100-199
        "Building",   # 200-299
        "Bus",        # 300-399
        "Dinosaur",   # 400-499
        "Elephant",   # 500-599
        "Flower",     # 600-699
        "Horse",      # 700-799
        "Mountain",   # 800-899
        "Food",       # 900-999
    ]


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

    # Menggunakan nama kelas yang sudah diperbarui
    class_names = _get_class_names()

    # Plot confusion matrix
    utils.ensure_directories()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8)) # Sedikit lebih besar untuk nama kelas
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
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
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
    Modifikasi: Mencari sampel dari kelas yang berbeda.

    - Menampilkan grid gambar (misal 3x3)
    - Title tiap gambar:
        T: true_class / P: pred_class
        top-3: class (prob%)

    Parameter
    ---------
    model : tf.keras.Model
        Model terlatih.
    num_examples : int
        Jumlah contoh yang divisualisasikan (maks 10).
    save_path : Path
        Lokasi file PNG. Jika None, simpan sebagai 'sample_predictions_test.png'
        di folder figures.
    """
    if save_path is None:
        save_path = Path(config.FIGURE_DIR) / "sample_predictions_test.png"
    save_path = Path(save_path)
    
    # Batasi num_examples agar tidak melebihi jumlah kelas
    if num_examples > config.NUM_CLASSES:
        print(f"Warning: num_examples dibatasi ke {config.NUM_CLASSES} (jumlah kelas).")
        num_examples = config.NUM_CLASSES

    print(f"==> Visualizing {num_examples} sample predictions (mencari kelas berbeda)...")
    test_ds = dataset.get_test_dataset(batch_size=config.BATCH_SIZE)

    # --- PERUBAHAN DI SINI ---
    # unbatch dan cari 1 sampel per kelas, sampai num_examples terpenuhi
    test_unbatched = test_ds.unbatch()

    images_list = []
    labels_list = []
    
    # {class_id: (img_tensor, label_tensor)}
    samples_found: Dict[int, Tuple[np.ndarray, np.ndarray]] = {} 

    # Iterasi dataset untuk mencari 1 sampel per kelas
    for img, label in test_unbatched:
        class_id = int(tf.argmax(label).numpy())
        
        # Jika kelas ini belum ditemukan, simpan
        if class_id not in samples_found:
            samples_found[class_id] = (img.numpy(), label.numpy())
        
        # Jika sudah terkumpul N sampel (sesuai num_examples)
        if len(samples_found) >= num_examples:
            break

    # Urutkan berdasarkan class_id agar tampilannya konsisten
    # dan ambil hanya num_examples pertama jika kita menemukan lebih
    sorted_keys = sorted(samples_found.keys())[:num_examples]
    for i in sorted_keys:
        images_list.append(samples_found[i][0])
        labels_list.append(samples_found[i][1])
    # --- AKHIR PERUBAHAN LOGIKA PENGAMBILAN GAMBAR ---

    if not images_list:
        print("Tidak ada data di test set untuk divisualisasikan.")
        return

    images_arr = np.stack(images_list, axis=0)
    labels_arr = np.stack(labels_list, axis=0)

    # prediksi
    probs = model.predict(images_arr, verbose=0)
    true_classes = np.argmax(labels_arr, axis=1)
    pred_classes = np.argmax(probs, axis=1)

    # Menggunakan nama kelas yang sudah diperbarui
    class_names = _get_class_names()

    # grid: sebisa mungkin square (misal 3x3 untuk 9 gambar)
    n = len(images_arr)
    grid_size = int(np.ceil(np.sqrt(n)))

    utils.ensure_directories()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3 * grid_size, 3.5 * grid_size)) # Tambah tinggi untuk title

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
        top3_info = "\n".join( # Ganti jadi newline
            [f"- {class_names[j]} ({probs[i][j]*100:.1f}%)" for j in top3_idx]
        )

        title = f"True: {class_names[true_id]}\nPred: {class_names[pred_id]}\n---\n{top3_info}"
        
        # Beri warna title
        color = "green" if true_id == pred_id else "red"
        ax.set_title(title, fontsize=8, color=color, loc='left')

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
    visualize_sample_predictions(model, num_examples=9) # Ambil 9 kelas berbeda

    print("\nRingkasan:")
    print(f"- Test loss      : {test_loss:.4f}")
    print(f"- Test accuracy  : {test_acc:.4f}")
    print(f"- Error ratio    : {error_ratio:.4f}")
    print(f"- Confusion matrix shape: {cm.shape}")


if __name__ == "__main__":
    # Jalankan dari root project:
    # $ python src/evaluate.py
    # atau
    # $ python -m src.evaluate
    main()