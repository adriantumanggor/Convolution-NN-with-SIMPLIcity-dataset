"""
config.py

Menyimpan konfigurasi global untuk proyek SIMPLIcity CNN:
- path dataset dan output
- hyperparameter training
- pengaturan gambar (ukuran, channel)
"""

from pathlib import Path

# Root project: simplicity_cnn/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# PATH DATASET
# ---------------------------------------------------------------------------

# Asumsi: sudah ada symlink:
# simplicity_cnn/data/SIMPLIcity -> ~/github/neuralcomp/SIMPLIcity Dataset
DATA_ROOT = PROJECT_ROOT / "data" / "SIMPLIcity"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"

# ---------------------------------------------------------------------------
# PATH OUTPUT
# ---------------------------------------------------------------------------

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_ROOT / "logs"
MODEL_DIR = OUTPUT_ROOT / "models"
FIGURE_DIR = OUTPUT_ROOT / "figures"

# Nama model utama
MODEL_NAME = "simplicity_cnn_tf"

BEST_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_best.h5"
HISTORY_CSV_PATH = LOG_DIR / "train_history.csv"
HISTORY_JSON_PATH = LOG_DIR / "train_history.json"

LOSS_FIG_PATH = FIGURE_DIR / "loss.png"
ACC_FIG_PATH = FIGURE_DIR / "accuracy.png"
CONF_MAT_FIG_PATH = FIGURE_DIR / "confusion_matrix_test.png"

# ---------------------------------------------------------------------------
# PENGATURAN GAMBAR
# ---------------------------------------------------------------------------

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Spesifikasi resize + padding sesuai instruksi:
# - inner side ~150
# - padding 37 di dua sisi (atas-bawah atau kiri-kanan)
RESIZED_MIN_SIDE = 150
PADDING_SIZE = 37

# ---------------------------------------------------------------------------
# HYPERPARAMETER TRAINING
# ---------------------------------------------------------------------------

NUM_CLASSES = 10        # SIMPLIcity: 10 kategori
BATCH_SIZE = 16
EPOCHS = 50

LEARNING_RATE = 1e-3

# Early stopping
EARLY_STOPPING_MONITOR = "val_loss"
EARLY_STOPPING_PATIENCE = 6  # di antara 5–7 sesuai instruksi

# Learning rate decay (exponential decay)
LR_DECAY_RATE = 0.9
LR_DECAY_STEPS = 5      # bisa diartikan "setiap 5 epoch"

# ---------------------------------------------------------------------------
# RANDOM SEED (untuk reproducibility)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
