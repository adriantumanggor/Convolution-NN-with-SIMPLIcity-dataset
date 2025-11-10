"""
dataset.py

Menyediakan loader dataset untuk SIMPLIcity dengan struktur:
data/SIMPLIcity/
  ├── train/
  │   ├── 0.jpg
  │   ├── 1.jpg
  │   └── ...
  ├── validation/
  │   ├── 150.jpg
  │   └── ...
  └── test/
      ├── 180.jpg
      └── ...

Asumsi label:
- Nama file: k.jpg, dengan k integer 0..999
- Label kelas: k // 100  (0-99 -> class 0, 100-199 -> class 1, ..., 900-999 -> class 9)

Preprocessing:
- Resize + padding ke 224x224:
  - Landscape (width > height):
      resize ke (height=150, width=224)
      lalu pad: 37 px di atas dan 37 px di bawah
  - Portrait (height > width):
      resize ke (height=224, width=150)
      lalu pad: 37 px di kiri dan 37 px di kanan
  - Square-ish (height == width):
      resize langsung ke 224x224 tanpa padding
- Normalisasi: float32, nilai 0-1 (img / 255.)

Augmentasi (hanya train):
- Random horizontal flip
- Random rotation ± ~15 derajat
- Random brightness ±10%
"""

import os
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

from . import config

AUTOTUNE = tf.data.AUTOTUNE


# -----------------------------------------------------------------------------
# Utility: list file gambar per split
# -----------------------------------------------------------------------------

def _list_image_files(split_dir: Path) -> List[str]:
    """
    Mengambil semua path file gambar di direktori split (train/val/test)
    dalam bentuk list string, disortir.

    Mencari ekstensi: .jpg, .jpeg, .png (case-sensitive & insensitive).
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(split_dir.glob(ext)))

    if not files:
        raise FileNotFoundError(
            f"Tidak ditemukan file gambar di direktori: {split_dir}"
        )

    return [str(f) for f in files]


# -----------------------------------------------------------------------------
# Label dari nama file
# -----------------------------------------------------------------------------

def _parse_label_from_filename(path: tf.Tensor) -> tf.Tensor:
    """
    Menghasilkan label one-hot dari path file.

    Asumsi:
    - path berisi ".../<split>/<k>.jpg"
    - k adalah integer 0..999
    - kelas = k // 100, dibatasi 0..NUM_CLASSES-1

    Return:
    - tensor float32 shape (NUM_CLASSES,)
    """
    # path: string scalar tensor
    # ambil nama file
    filename = tf.strings.split(path, os.sep)[-1]            # "123.jpg"

    # buang ekstensi
    stem = tf.strings.regex_replace(filename, r"\.[^\.]+$", "")  # "123"

    # konversi ke integer
    index = tf.strings.to_number(stem, out_type=tf.int32)    # 123

    # kelas = index // 100
    class_id = index // 100

    # batasi di range 0..NUM_CLASSES-1, just in case
    class_id = tf.clip_by_value(
        class_id, 0, config.NUM_CLASSES - 1
    )

    # one-hot
    label = tf.one_hot(class_id, depth=config.NUM_CLASSES, dtype=tf.float32)
    return label


# -----------------------------------------------------------------------------
# Resize + padding 224x224 sesuai spesifikasi
# -----------------------------------------------------------------------------

def _resize_and_pad_image(img: tf.Tensor) -> tf.Tensor:
    """
    Melakukan resize + padding ke ukuran final 224x224x3.

    Aturan:
    - Jika width > height (landscape):
        resize -> (height=150, width=224)
        pad top & bottom dengan 37 px (warna hitam)
    - Jika height > width (portrait):
        resize -> (height=224, width=150)
        pad left & right dengan 37 px
    - Jika height == width:
        resize langsung -> (224, 224)
    """
    img_height = tf.shape(img)[0]
    img_width = tf.shape(img)[1]

    def _landscape():
        # (H, W) -> (150, 224)
        resized = tf.image.resize(
            img,
            size=(config.RESIZED_MIN_SIDE, config.IMG_WIDTH),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        # pad 37 px di atas & bawah
        paddings = tf.constant([
            [config.PADDING_SIZE, config.PADDING_SIZE],  # height: top, bottom
            [0, 0],                                     # width
            [0, 0],                                     # channel
        ])
        padded = tf.pad(resized, paddings, mode="CONSTANT", constant_values=0.0)
        return padded

    def _portrait():
        # (H, W) -> (224, 150)
        resized = tf.image.resize(
            img,
            size=(config.IMG_HEIGHT, config.RESIZED_MIN_SIDE),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        # pad 37 px di kiri & kanan
        paddings = tf.constant([
            [0, 0],                                     # height
            [config.PADDING_SIZE, config.PADDING_SIZE],  # width: left, right
            [0, 0],                                     # channel
        ])
        padded = tf.pad(resized, paddings, mode="CONSTANT", constant_values=0.0)
        return padded

    def _square():
        # langsung resize ke 224x224
        return tf.image.resize(
            img,
            size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            method=tf.image.ResizeMethod.BILINEAR,
        )

    img = tf.cond(
        img_width > img_height,
        true_fn=_landscape,
        false_fn=lambda: tf.cond(
            img_height > img_width,
            true_fn=_portrait,
            false_fn=_square,
        ),
    )

    # Untuk jaga-jaga jika ada rounding effect, crop/pad ke 224x224 fix
    img = tf.image.resize_with_crop_or_pad(img, config.IMG_HEIGHT, config.IMG_WIDTH)
    return img


# -----------------------------------------------------------------------------
# Augmentasi (hanya untuk train)
# -----------------------------------------------------------------------------

# Layer RandomRotation untuk digunakan dalam augmentasi.
_RANDOM_ROT_LAYER = tf.keras.layers.RandomRotation(factor=0.08)  # ~±15 derajat


def _augment_image(img: tf.Tensor) -> tf.Tensor:
    """
    Menerapkan augmentasi ringan pada gambar:
    - random horizontal flip
    - random rotation (± ~15 derajat)
    - random brightness ±10%
    """
    # random horizontal flip
    img = tf.image.random_flip_left_right(img)

    # random rotation via Keras layer (butuh dimensi batch)
    img_batched = tf.expand_dims(img, axis=0)
    img_batched = _RANDOM_ROT_LAYER(img_batched, training=True)
    img = tf.squeeze(img_batched, axis=0)

    # random brightness
    delta = tf.random.uniform([], minval=-0.1, maxval=0.1)
    img = tf.image.adjust_brightness(img, delta)
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img


# -----------------------------------------------------------------------------
# Load & preprocess full pipeline
# -----------------------------------------------------------------------------

def _load_and_preprocess_image(path: tf.Tensor, augment: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Membaca file gambar dari path, melakukan decode, resize+padding, normalisasi,
    dan (opsional) augmentasi.

    Return:
    - image: float32 tensor shape (224,224,3)
    - label: one-hot float32 tensor shape (NUM_CLASSES,)
    """
    # baca file
    image_bytes = tf.io.read_file(path)
    # decode JPEG (dipaksa 3 channel RGB)
    image = tf.io.decode_jpeg(image_bytes, channels=config.IMG_CHANNELS)

    # konversi ke float32, skala 0..1 (otomatis /255)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # resize + padding 224x224
    image = _resize_and_pad_image(image)

    # augmentasi jika diminta (train)
    if augment:
        image = _augment_image(image)

    # label dari nama file
    label = _parse_label_from_filename(path)

    return image, label


def _build_dataset(file_paths: List[str],
                   batch_size: int,
                   shuffle: bool,
                   augment: bool) -> tf.data.Dataset:
    """
    Membangun tf.data.Dataset dari list path file.

    Parameter
    ---------
    file_paths : List[str]
        Daftar path file gambar.
    batch_size : int
        Ukuran batch.
    shuffle : bool
        Apakah dataset di-shuffle setiap epoch.
    augment : bool
        Apakah augmentasi diterapkan.

    Return
    ------
    tf.data.Dataset
        Dataset (image, label) siap untuk model.fit().
    """
    ds = tf.data.Dataset.from_tensor_slices(file_paths)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p: _load_and_preprocess_image(p, augment=augment),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


# -----------------------------------------------------------------------------
# Public API: fungsi untuk mendapatkan train/val/test dataset
# -----------------------------------------------------------------------------

def get_train_dataset(batch_size: int = None) -> tf.data.Dataset:
    """
    Mengembalikan tf.data.Dataset untuk split train.

    Augmentasi: AKTIF (flip, rotasi, brightness).

    Parameter
    ---------
    batch_size : int
        Ukuran batch. Jika None, menggunakan config.BATCH_SIZE.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    file_paths = _list_image_files(config.TRAIN_DIR)
    ds = _build_dataset(
        file_paths=file_paths,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
    )
    return ds


def get_validation_dataset(batch_size: int = None) -> tf.data.Dataset:
    """
    Mengembalikan tf.data.Dataset untuk split validation.

    Augmentasi: NONAKTIF (hanya resize+padding+normalisasi).
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    file_paths = _list_image_files(config.VAL_DIR)
    ds = _build_dataset(
        file_paths=file_paths,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    return ds


def get_test_dataset(batch_size: int = None) -> tf.data.Dataset:
    """
    Mengembalikan tf.data.Dataset untuk split test.

    Augmentasi: NONAKTIF (hanya resize+padding+normalisasi).
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    file_paths = _list_image_files(config.TEST_DIR)
    ds = _build_dataset(
        file_paths=file_paths,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    return ds


def get_datasets(batch_size: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Helper untuk mendapatkan (train_ds, val_ds, test_ds) sekaligus.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds = get_train_dataset(batch_size=batch_size)
    val_ds = get_validation_dataset(batch_size=batch_size)
    test_ds = get_test_dataset(batch_size=batch_size)

    return train_ds, val_ds, test_ds
