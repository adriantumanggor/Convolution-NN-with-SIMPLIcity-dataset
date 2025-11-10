"""
train.py

Script utama untuk melatih model CNN pada dataset SIMPLIcity.

Fitur:
- Menggunakan tf.data.Dataset dari dataset.py
- Optimizer: Adam + ExponentialDecay learning rate schedule
- Loss: Categorical Crossentropy
- Metric: Accuracy
- Callbacks:
    - EarlyStopping (monitor val_loss)
    - ModelCheckpoint (save best model berdasarkan val_accuracy)
- Logging:
    - Simpan history ke CSV dan JSON
    - Plot loss & accuracy ke PNG
    - Simpan model terakhir (last) dan model terbaik (best)
"""

from pathlib import Path

import tensorflow as tf

from . import config
from . import dataset
from . import model as model_module
from . import utils


def build_optimizer_with_schedule() -> tf.keras.optimizers.Optimizer:
    """
    Membangun optimizer Adam dengan ExponentialDecay learning rate schedule.

    - initial_lr: config.LEARNING_RATE
    - decay_rate: config.LR_DECAY_RATE (misal 0.9)
    - decay_steps: kira-kira setiap beberapa epoch.
      Di sini kita set "per-epoch-like" dengan asumsi step per epoch ~1000.
      Untuk praktikum, ini cukup; kalau mau akurat bisa disesuaikan
      dengan jumlah batch train sebenarnya.
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=1000,              # heuristik sederhana
        decay_rate=config.LR_DECAY_RATE,
        staircase=True,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer


def build_callbacks() -> list:
    """
    Membangun list callbacks Keras:
    - EarlyStopping (val_loss)
    - ModelCheckpoint (save best val_accuracy)
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.BEST_MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    # Bisa ditambah TensorBoard / CSVLogger jika mau, tapi
    # kita sudah menyimpan history manual via utils.save_history.
    return [early_stop, checkpoint]


def compile_model() -> tf.keras.Model:
    """
    Membuat dan meng-compile model CNN sesuai spesifikasi.

    - Optimizer: Adam + ExponentialDecay
    - Loss: categorical_crossentropy
    - Metrics: accuracy
    """
    model = model_module.build_model(name=config.MODEL_NAME)

    optimizer = build_optimizer_with_schedule()

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train() -> None:
    """
    Fungsi utama training:

    1. Set random seed & pastikan direktori output ada.
    2. Bangun tf.data.Dataset untuk train & validation.
    3. Bangun dan compile model CNN.
    4. Jalankan model.fit dengan callbacks.
    5. Simpan history (CSV + JSON) dan plot loss & accuracy.
    6. Simpan model terakhir (last model).
    """
    # 1. Setup environment (seed + folder output)
    utils.set_seed()
    utils.ensure_directories()

    print("==> Loading datasets...")
    train_ds = dataset.get_train_dataset(batch_size=config.BATCH_SIZE)
    val_ds = dataset.get_validation_dataset(batch_size=config.BATCH_SIZE)

    print("==> Building and compiling model...")
    model = compile_model()
    model.summary()

    callbacks = build_callbacks()

    print("==> Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("==> Saving training history (CSV + JSON)...")
    utils.save_history(history)

    print("==> Plotting training curves (loss & accuracy)...")
    utils.plot_training_curves(history)

    # Simpan model terakhir (bukan necessarily best)
    last_model_path = Path(config.MODEL_DIR) / f"{config.MODEL_NAME}_last.h5"
    last_model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"==> Saving last model to: {last_model_path}")
    model.save(str(last_model_path))

    print("==> Training selesai.")


if __name__ == "__main__":
    # Jalankan training ketika file ini dieksekusi langsung:
    # $ python -m src.train
    # atau dari root project:
    # $ python src/train.py
    train()
