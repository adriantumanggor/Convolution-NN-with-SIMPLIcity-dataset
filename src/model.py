"""
model.py

Mendefinisikan arsitektur CNN untuk klasifikasi gambar SIMPLIcity.

Spesifikasi arsitektur (sesuai instruksi):

Input: 224 x 224 x 3

Conv Block 1:
- Conv2D: 16 filters, kernel 3x3, stride 1, padding 'same'
- Batch Normalization
- ReLU

Conv Block 2:
- Conv2D: 32 filters, kernel 3x3, stride 1, padding 'same'
- Batch Normalization
- ReLU

Conv Block 3:
- Conv2D: 64 filters, kernel 3x3, stride 1, padding 'same'
- Batch Normalization
- ReLU

Global Average Pooling:
- Mengubah feature map (224 x 224 x 64) menjadi vektor length 64

Classifier:
- Dropout (p=0.5)
- Dense: 10 units, Softmax (10 kelas SIMPLIcity)

Catatan:
- Kernel 3x3: standar VGG-style, cukup kuat untuk menangkap pola lokal.
- Filter naik 16 -> 32 -> 64: kapasitas naik dengan kedalaman.
- 'same' padding + stride 1: resolusi spasial tetap, cocok dengan GAP.
- BatchNorm + ReLU: stabilitas dan kecepatan konvergensi.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config


def build_model(name: str = "simplicity_cnn") -> tf.keras.Model:
    """
    Membangun model CNN Keras sesuai spesifikasi.

    Parameter
    ---------
    name : str
        Nama model Keras.

    Return
    ------
    tf.keras.Model
        Model yang BELUM di-compile (compile dilakukan di train.py).
    """
    inputs = layers.Input(
        shape=config.INPUT_SHAPE,
        name="input_image"
    )

    # ------------------------------------------------------------------
    # Conv Block 1
    # ------------------------------------------------------------------
    # 3x3 conv, 16 filter, padding 'same', tanpa bias (karena ada BatchNorm)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name="conv_block1_conv",
    )(inputs)
    x = layers.BatchNormalization(name="conv_block1_bn")(x)
    x = layers.ReLU(name="conv_block1_relu")(x)

    # ------------------------------------------------------------------
    # Conv Block 2
    # ------------------------------------------------------------------
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name="conv_block2_conv",
    )(x)
    x = layers.BatchNormalization(name="conv_block2_bn")(x)
    x = layers.ReLU(name="conv_block2_relu")(x)

    # ------------------------------------------------------------------
    # Conv Block 3
    # ------------------------------------------------------------------
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name="conv_block3_conv",
    )(x)
    x = layers.BatchNormalization(name="conv_block3_bn")(x)
    x = layers.ReLU(name="conv_block3_relu")(x)

    # ------------------------------------------------------------------
    # Global Average Pooling + Dropout + Classifier
    # ------------------------------------------------------------------
    # GAP mereduksi feature map besar menjadi vektor 64 dimensi
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # Dropout untuk mengurangi overfitting:
    # - dataset relatif kecil (1000 gambar)
    # - parameter fully-connected relatif sedikit (karena GAP)
    x = layers.Dropout(rate=0.5, name="dropout")(x)

    outputs = layers.Dense(
        units=config.NUM_CLASSES,
        activation="softmax",
        name="predictions",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=name,
    )

    return model


def build_and_compile_model(
    name: str = "simplicity_cnn",
    learning_rate: Optional[float] = None,
) -> tf.keras.Model:
    """
    Opsional helper: membangun & meng-compile model dengan Adam + Categorical Crossentropy.
    Compile utama sebaiknya tetap diatur di train.py, tapi fungsi ini bisa
    dipakai di notebook untuk eksperimen cepat.

    Parameter
    ---------
    name : str
        Nama model.
    learning_rate : float atau None
        Jika None, gunakan config.LEARNING_RATE.

    Return
    ------
    tf.keras.Model
        Model yang sudah di-compile.
    """
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    model = build_model(name=name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
