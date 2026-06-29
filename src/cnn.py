import os
import numpy as np

IMG_SIZE = 64
EPOCHS   = 15
BATCH    = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "cnn_model.h5")


# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────

def get_labels(dataset_dir):
    """Return sorted list of sub-folder names (= class labels)."""
    return sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])


def load_dataset(dataset_dir):
    """Load all images from dataset_dir/<label>/ sub-folders.

    Returns
    -------
    X      : np.ndarray  shape (N, IMG_SIZE, IMG_SIZE, 3), float32 0-1
    y      : np.ndarray  shape (N,), int  class indices
    labels : list[str]   class names in alphabetical order
    """
    import cv2

    labels = get_labels(dataset_dir)
    if len(labels) < 2:
        raise ValueError(
            f"Dataset harus punya minimal 2 sub-folder (class). "
            f"Ditemukan: {labels or 'tidak ada'}"
        )

    X, y = [], []
    for idx, label in enumerate(labels):
        folder = os.path.join(dataset_dir, label)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            print(f"[WARNING] Folder '{label}' kosong, dilewati.")
            continue
        for fname in files:
            path = os.path.join(folder, fname)
            img  = cv2.imread(path)
            if img is None:
                print(f"[WARNING] Gagal baca: {path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(idx)

    if len(X) == 0:
        raise ValueError("Tidak ada gambar yang berhasil dimuat dari dataset.")

    return np.array(X, dtype=np.float32) / 255.0, np.array(y), labels


# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────

def build_model(num_classes):
    """3-Conv CNN sederhana. Tambah layer jika val_accuracy < 70%."""
    from tensorflow.keras import layers, models

    m = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    m.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(dataset_dir, callback=None):
    """Train model dan simpan ke MODEL_PATH.

    Parameters
    ----------
    dataset_dir : str   path ke folder dataset
    callback    : callable(epoch, total, acc, val_acc) | None
                  dipanggil tiap akhir epoch untuk update progress bar GUI
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import Callback

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X, y, labels = load_dataset(dataset_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(len(labels))

    class _ProgressCB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if callback:
                callback(
                    epoch + 1, EPOCHS,
                    logs.get("accuracy", 0.0),
                    logs.get("val_accuracy", 0.0),
                )

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_data=(X_val, y_val),
        callbacks=[_ProgressCB()],
        verbose=0,
    )
    model.save(MODEL_PATH)
    return model, labels


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict(img_rgb, labels):
    """Prediksi satu gambar (RGB numpy array).

    Returns
    -------
    label      : str    nama kelas terprediksi
    confidence : float  probabilitas 0–1
    """
    import cv2
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    img   = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    pred  = model.predict(np.expand_dims(img, axis=0), verbose=0)
    idx   = int(np.argmax(pred))
    return labels[idx], float(pred[0][idx])


def model_exists():
    """Cek apakah model sudah tersimpan."""
    return os.path.exists(MODEL_PATH)