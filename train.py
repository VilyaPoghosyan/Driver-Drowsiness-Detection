import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
 

tf.random.set_seed(42)
np.random.seed(42)
 
IMG_SIZE   = 80
BATCH_SIZE = 32
EPOCHS     = 20
 
 
def build_data_generators(data_dir: str, batch_size: int):
    """
    Build train/validation data generators.
 
    Preprocessing applied (identical pipeline used at inference):
      - Resize images to 80×80 pixels
      - Normalize pixel values to [0, 1]
      - 80/20 train-val split
 
    Augmentation (train only — to improve generalization):
      - Random horizontal flip
      - ±15° rotation
      - ±10% zoom
      - ±10% width/height shift
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
    )
 
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )
 
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )
 
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )
 
    return train_gen, val_gen
 
 
def build_baseline_model(input_shape=(80, 80, 3)) -> tf.keras.Model:
    """
    BASELINE MODEL — simple 2-layer CNN.
    No augmentation, no batch norm, no dropout.
    Purpose: establish a lower-bound accuracy to compare against the final model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
 
        
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
 
        # Block 2
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
 
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),   
    ], name="baseline_cnn")
    return model
 
 
def build_final_model(input_shape=(80, 80, 3)) -> tf.keras.Model:
    """
    FINAL MODEL — deeper CNN with BatchNormalization and Dropout.
 
    Architecture decisions:
      - 3 convolutional blocks (16 → 32 → 64 filters) to capture low-level
        texture (lip edges) and higher-level shape features (open mouth).
      - BatchNormalization after each conv block to stabilize training and
        allow higher learning rates.
      - Dropout(0.4) before the dense head to reduce overfitting on the
        relatively small yawn dataset.
      - Binary sigmoid output: score > 0.5 = yawning.
 
    Loss:   Binary Cross-Entropy (standard for binary classification)
    Optimizer: Adam (lr=1e-3, adaptive — well-suited for CNNs)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
 
        
        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),                      
        layers.Dense(1, activation="sigmoid"),
    ], name="final_cnn")
    return model
 

 
def train_model(model, train_gen, val_gen, epochs: int, model_name: str):
    """Compile, train and return (model, history)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
 
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_name}_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
 
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history
 

 
def plot_training_curves(baseline_hist, final_hist, save_path="results/training_curves.png"):
    """Plot train/val accuracy and loss curves for both models."""
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves — Baseline vs Final CNN", fontsize=14)
 
    for col, (label, hist) in enumerate(
        [("Baseline CNN", baseline_hist), ("Final CNN", final_hist)]
    ):
        # Accuracy
        axes[0][col].plot(hist.history["accuracy"],     label="Train Acc")
        axes[0][col].plot(hist.history["val_accuracy"], label="Val Acc")
        axes[0][col].set_title(f"{label} — Accuracy")
        axes[0][col].set_xlabel("Epoch")
        axes[0][col].legend()
 
        # Loss
        axes[1][col].plot(hist.history["loss"],     label="Train Loss")
        axes[1][col].plot(hist.history["val_loss"], label="Val Loss")
        axes[1][col].set_title(f"{label} — Loss")
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].legend()
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Training curves saved to {save_path}")
    plt.close()
 
 
def evaluate_and_plot_confusion(model, val_gen, model_name: str):
    """Print classification report and save a confusion matrix PNG."""
    os.makedirs("results", exist_ok=True)
 
    val_gen.reset()
    y_true = val_gen.classes
 
    # Predict in batches
    preds = model.predict(val_gen, verbose=0)
    y_pred = (preds.flatten() > 0.5).astype(int)
 
    class_names = list(val_gen.class_indices.keys())
    print(f"\n── {model_name} Classification Report ──")
    print(classification_report(y_true, y_pred, target_names=class_names))
 
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    save_path = f"results/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.close()
 

 
def main():
    parser = argparse.ArgumentParser(description="Train mouth-yawn CNN classifier")
    parser.add_argument("--data_dir",   type=str,  default="data/",  help="Root dataset directory")
    parser.add_argument("--epochs",     type=int,  default=EPOCHS,   help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int,  default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
 
    print(f"[INFO] Dataset : {args.data_dir}")
    print(f"[INFO] Epochs  : {args.epochs}")
    print(f"[INFO] Batch   : {args.batch_size}")
 
   
    train_gen, val_gen = build_data_generators(args.data_dir, args.batch_size)
    print(f"[INFO] Classes : {train_gen.class_indices}")
 
    print("\n" + "═" * 60)
    print("  STEP 1: TRAINING BASELINE MODEL")
    print("═" * 60)
    baseline = build_baseline_model()
    baseline, baseline_hist = train_model(
        baseline, train_gen, val_gen, args.epochs, "baseline_cnn"
    )
    evaluate_and_plot_confusion(baseline, val_gen, "Baseline CNN")
 
    
    print("\n" + "═" * 60)
    print("  STEP 2: TRAINING FINAL MODEL")
    print("═" * 60)
    final = build_final_model()
    final, final_hist = train_model(
        final, train_gen, val_gen, args.epochs, "mouth_cnn"
    )
    evaluate_and_plot_confusion(final, val_gen, "Final CNN")
 
   
    final.save("mouth_cnn.h5")
    print("[INFO] Final model saved to mouth_cnn.h5")
 
    
    plot_training_curves(baseline_hist, final_hist)
 
    print("\n[DONE] Training complete. Files saved:")
    print("  mouth_cnn.h5               ← production model for inference")
    print("  baseline_cnn_best.h5       ← baseline weights")
    print("  results/training_curves.png")
    print("  results/confusion_matrix_*.png")
 
 
if __name__ == "__main__":
    main()
 