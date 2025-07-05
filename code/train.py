import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import seaborn as sns

# ==============
# Configuration
# ==============
BASE_DIR = "YOUR_PROJECT_PATH"
TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
VAL_DIR = os.path.join(BASE_DIR, "data/val")
TEST_DIR = os.path.join(BASE_DIR, "data/test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
MIN_DELTA = 0.001
INITIAL_LR = 0.001
TTA_STEPS = 5  # For test-time augmentation

# =================
# Data Preparation
# =================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    brightness_range=[0.95, 1.05]
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# ===================
# Model Architecture
# ===================
def build_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Save model architecture as PNG
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(PLOT_DIR, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )
    return model


model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3), train_generator.num_classes)

# ==========
# Callbacks
# ==========
training_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# =================
# Initial Training
# =================
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== Phase 1: Initial Training ===")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=training_callbacks,
    verbose=1
)

# Save initial model
model.save(os.path.join(MODEL_DIR, 'initial_model.keras'))

# ========================
# Fine-Tuning (Last Push)
# ========================
def fine_tune_model(model):
    # Unfreeze last 8 layers of base model
    for layer in model.layers[0].layers[-8:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # 10x lower than initial
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== Phase 2: Fine-Tuning ===")
    history_ft = model.fit(
        train_generator,
        epochs=10,  # Short fine-tuning phase
        validation_data=val_generator,
        callbacks=[
            EarlyStopping(
                monitor='val_accuracy',
                patience=2,  # Aggressive for fine-tuning
                restore_best_weights=True,
                verbose=1
            )
        ],
        verbose=1
    )
    return model, history_ft

# Load best model and fine-tune
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
model, history_ft = fine_tune_model(model)
model.save(os.path.join(MODEL_DIR, 'fine_tuned_model.keras'))

# =======================
# Test-Time Augmentation
# =======================
def evaluate_with_tta(model, generator, steps=TTA_STEPS):
    tta_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=2,
        width_shift_range=0.02,
        height_shift_range=0.02,
        brightness_range=[0.98, 1.02]
    )
    
    tta_generator = tta_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    tta_predictions = []
    for _ in range(steps):
        tta_generator.reset()
        tta_predictions.append(model.predict(tta_generator))
    
    # Average predictions
    final_preds = np.mean(tta_predictions, axis=0).argmax(axis=1)
    test_acc = np.mean(final_preds == tta_generator.classes)
    
    return test_acc, final_preds, tta_generator.classes

print("\n=== Phase 3: TTA Evaluation ===")
tta_accuracy, tta_preds, tta_true = evaluate_with_tta(model, test_generator)
print(f"TTA Test Accuracy: {tta_accuracy*100:.2f}%")

# ======================
# Hard Example Mining
# ======================
def analyze_hard_examples(true_labels, pred_labels, filenames, class_names):
    misclassified = np.where(pred_labels != true_labels)[0]
    hard_examples = []
    
    for idx in misclassified:
        hard_examples.append({
            'filename': filenames[idx],
            'actual': class_names[true_labels[idx]],
            'predicted': class_names[pred_labels[idx]]
        })
    
    # Count most confused pairs
    confusion_pairs = Counter([(ex['actual'], ex['predicted']) for ex in hard_examples])
    print("\nTop 3 Confused Pairs:")
    for pair, count in confusion_pairs.most_common(3):
        print(f"{pair[0]} → {pair[1]}: {count} examples")
    
    return hard_examples, confusion_pairs

class_names = list(test_generator.class_indices.keys())
hard_examples, confusion_pairs = analyze_hard_examples(
    tta_true, tta_preds, test_generator.filenames, class_names
)

# Save hard examples
hard_examples_df = pd.DataFrame(hard_examples)
hard_examples_df.to_csv(os.path.join(RESULTS_DIR, 'hard_examples.csv'), index=False)

# ===========================
# Evaluation & Visualization
# ===========================
def save_confusion_matrix_plot(cm, class_names, filename):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_class_distribution_plot(generator, class_names, filename):
    class_counts = dict(zip(class_names, np.bincount(generator.classes)))
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_evaluation_results(model, generator, results_dir, tta=False):
    generator.reset()
    y_true = generator.classes
    y_pred = model.predict(generator).argmax(axis=1) if not tta else tta_preds
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(results_dir, 'tta_classification_report.csv' if tta else 'classification_report.csv'))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    pd.DataFrame(conf_matrix,
                index=class_names,
                columns=class_names
               ).to_csv(os.path.join(results_dir, 'tta_confusion_matrix.csv' if tta else 'confusion_matrix.csv'))
    
    # Save confusion matrix plot
    cm_filename = os.path.join(PLOT_DIR, 'tta_confusion_matrix.png' if tta else 'confusion_matrix.png')
    save_confusion_matrix_plot(conf_matrix, class_names, cm_filename)
    
    return model.evaluate(generator, return_dict=True)

# Save class distribution plots
save_class_distribution_plot(train_generator, class_names, os.path.join(PLOT_DIR, 'train_class_distribution.png'))
save_class_distribution_plot(val_generator, class_names, os.path.join(PLOT_DIR, 'val_class_distribution.png'))
save_class_distribution_plot(test_generator, class_names, os.path.join(PLOT_DIR, 'test_class_distribution.png'))

# Save standard evaluation
test_results = save_evaluation_results(model, test_generator, RESULTS_DIR)

# Save TTA evaluation
tta_results = save_evaluation_results(model, test_generator, RESULTS_DIR, tta=True)

# Enhanced plotting
def plot_training_history(history, history_ft=None):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    if history_ft:
        plt.plot(np.arange(len(history.history['accuracy']), 
                          len(history.history['accuracy'])+len(history_ft.history['accuracy'])), 
                history_ft.history['accuracy'], 'b--', label='FT Train Accuracy')
        plt.plot(np.arange(len(history.history['val_accuracy']), 
                          len(history.history['val_accuracy'])+len(history_ft.history['val_accuracy'])), 
                history_ft.history['val_accuracy'], 'r--', label='FT Val Accuracy')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'accuracy_curve.png'))
    plt.close()
    
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    if history_ft:
        plt.plot(np.arange(len(history.history['loss']), 
                          len(history.history['loss'])+len(history_ft.history['loss'])), 
                history_ft.history['loss'], 'b--', label='FT Train Loss')
        plt.plot(np.arange(len(history.history['val_loss']), 
                          len(history.history['val_loss'])+len(history_ft.history['val_loss'])), 
                history_ft.history['val_loss'], 'r--', label='FT Val Loss')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'loss_curve.png'))
    plt.close()

plot_training_history(history, history_ft)

# =============
# Final Report
# =============
print("\n=== Final Results ===")
print(f"Standard Test Accuracy: {test_results['accuracy']*100:.2f}%")
print(f"TTA Test Accuracy: {tta_accuracy*100:.2f}%")
print(f"Most Confused Pairs:")
for pair, count in confusion_pairs.most_common(3):
    print(f"  {pair[0]} → {pair[1]}: {count} misclassifications")

print("\nTraining pipeline completed!")
print(f"Models saved to: {MODEL_DIR}")
print(f"Plots saved to: {PLOT_DIR}")
print(f"Results saved to: {RESULTS_DIR}")