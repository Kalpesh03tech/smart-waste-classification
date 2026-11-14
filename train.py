import argparse
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning noise
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glass", "metal", "paper", "plastic"]  # Alphabetical — matches tf.keras.utils.image_dataset_from_directory default
NUM_CLASSES = len(CLASS_NAMES)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train waste classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing one subfolder per class")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Maximum training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate for the classifier head (default: 1e-3)")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5,
                        help="Learning rate for fine-tuning phase (default: 1e-5)")
    parser.add_argument("--fine_tune_epochs", type=int, default=10,
                        help="Additional epochs for fine-tuning (default: 10)")
    return parser.parse_args()

# Data loading
def load_datasets(data_dir: str, batch_size: int):
    # Step 1: 85/15 split → train_val / test
    train_val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        validation_split=0.15,
        subset="training",
    )

    test_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        validation_split=0.15,
        subset="validation",
    )

    # Step 2: Sub-split train_val → train / val 
    total_batches = tf.data.experimental.cardinality(train_val_ds).numpy()
    val_batches = max(1, int(total_batches * 0.176))
    train_batches = total_batches - val_batches

    # Cache dataset to ensure consistent deterministic partition between train and val
    cached_train_val = train_val_ds.cache()
    val_ds = cached_train_val.take(val_batches)
    train_ds = cached_train_val.skip(val_batches)

    print(f"Data split (batches): train={train_batches}, val={val_batches}, "
          f"test={tf.data.experimental.cardinality(test_ds).numpy()}")

    return train_ds, val_ds, test_ds, CLASS_NAMES


def compute_class_weights(data_dir: str):
    """Compute class weights inversely proportional to sample counts."""
    counts = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"WARNING: class folder '{cls_dir}' not found — check your dataset structure.")
            counts.append(0)
        else:
            n = len([f for f in os.listdir(cls_dir)
                     if os.path.isfile(os.path.join(cls_dir, f))])
            counts.append(n)
    counts = np.array(counts, dtype=np.float64)
    if counts.sum() == 0:
        sys.exit(f"ERROR: No images found in {data_dir}. Expected subfolders: {CLASS_NAMES}")
    total = counts.sum()
    weights = total / (NUM_CLASSES * counts + 1e-6)  # Avoid division by zero
    class_weight = {i: w for i, w in enumerate(weights)}
    print(f"\nClass counts : {dict(zip(CLASS_NAMES, counts.astype(int).tolist()))}")
    print(f"Class weights: {class_weight}\n")
    return class_weight

# Data augmentation (applied only during training)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Model construction
def build_model(trainable_base: bool = False):
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    if trainable_base:
        # Unfreeze only the last 30 layers for fine-tuning
        for layer in base.layers[:-30]:
            layer.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # MobileNetV2 expects pixels in [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = data_augmentation(x)
    x = base(x, training=False)  # keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Evaluation helpers
def evaluate_on_test(model, test_ds, class_names):
    """Print classification report and save confusion matrix."""
    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    test_acc = np.mean(y_true == y_pred)
    print(f"Overall test accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    return test_acc

# Main training loop
def main():
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: data directory '{args.data_dir}' does not exist.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Verify class subfolders exist
    for cls in CLASS_NAMES:
        cls_path = os.path.join(args.data_dir, cls)
        if not os.path.isdir(cls_path):
            sys.exit(
                f"ERROR: Expected class folder '{cls_path}' not found.\n"
                f"Dataset must contain subfolders: {CLASS_NAMES}"
            )

    # Load data
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    class_weight = compute_class_weights(args.data_dir)

    # Phase 1: Train classifier head
    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier head (base frozen)")
    print("=" * 60)
    model = build_model(trainable_base=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda x: print(x))

    callbacks_p1 = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks_p1,
    )

    # Phase 2: Fine-tune top layers of base
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("=" * 60)

    # Unfreeze top layers
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_layer = layer
            break

    if base_layer is not None:
        base_layer.trainable = True
        for layer in base_layer.layers[:-30]:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_epochs = args.epochs + args.fine_tune_epochs
    callbacks_p2 = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=args.epochs,
        epochs=total_epochs,
        class_weight=class_weight,
        callbacks=callbacks_p2,
    )

    # Evaluate on test set
    best_model = keras.models.load_model(MODEL_PATH)
    test_acc = evaluate_on_test(best_model, test_ds, class_names)

    # Save class names alongside the model for app.py to use
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nNote: Reported accuracy reflects performance on the held-out test split.")
    print("Real-world accuracy depends on image quality and similarity to training data.")

if __name__ == "__main__":
    main()