# =====================================
# 1. IMPORT LIBRARIES
# =====================================

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================
# 2. PARAMETERS
# =====================================

DATASET_PATH = r"C:\Users\LENOVO\Desktop\audio foregery\dataset\Training"
SAMPLE_RATE = 16000
DURATION = 2   # 🔥 reduced for speed
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 85

print("🚀 Script started")
print("Dataset path:", DATASET_PATH)
print("Folders:", os.listdir(DATASET_PATH))


# =====================================
# 3. LOAD & EXTRACT FEATURES
# =====================================

def extract_features(file_path):
    try:
        # 🔥 FIX: remove resampling (VERY IMPORTANT)
        audio, sr = librosa.load(file_path, sr=None)

        # convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # pad or cut
        if len(audio) < SAMPLES_PER_FILE:
            audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))
        else:
            audio = audio[:SAMPLES_PER_FILE]

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return mfcc

    except Exception as e:
        print("❌ Error in:", file_path)
        return None


X = []
y = []

# 🔥 FIX: correct folder names
classes = ['Real', 'Fake']

for label, category in enumerate(classes):
    folder = os.path.join(DATASET_PATH, category)

    print(f"\n📂 Loading {category} data...")

    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue

    for i, file in enumerate(os.listdir(folder)):

        # 🔥 progress print
        if i % 100 == 0:
            print(f"{category}: {i} files processed")

        file_path = os.path.join(folder, file)

        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y.append(label)

# convert to array
X = np.array(X)
y = np.array(y)

print("\n✅ Total samples loaded:", len(X))

# 🔥 safety check
if len(X) == 0:
    print("❌ No data loaded. Check dataset!")
    exit()


# =====================================
# 4. RESHAPE FOR CNN
# =====================================

X = X[..., np.newaxis]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# =====================================
# 5. BUILD MODEL
# =====================================

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =====================================
# 6. CLASS WEIGHTS
# =====================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


# =====================================
# 7. TRAIN MODEL
# =====================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights
)


# =====================================
# 8. SAVE MODEL
# =====================================

model.save("audio_model_v1.keras")
print("✅ Model saved successfully!")

# =====================================
# 10. CONFUSION MATRIX
# =====================================

pred_probs = model.predict(X_val)
pred_labels = (pred_probs.flatten() > 0.5).astype(int)

cm = confusion_matrix(y_val, pred_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("graph_audio/confusion_matrix.png")
plt.show()


# =====================================
# 11. ACCURACY GRAPH
# =====================================

plt.figure()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epochs')

plt.savefig("graph_audio/accuracy.png")
plt.show()


# =====================================
# 12. LOSS GRAPH
# =====================================

plt.figure()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')

plt.savefig("graph_audio/loss.png")
plt.show()

# =====================================
# 10. CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:")
print(classification_report(y_val, pred_labels, target_names=classes))