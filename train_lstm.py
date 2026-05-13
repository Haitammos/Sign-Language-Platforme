"""
=====================================================
  TRAIN LSTM MODEL ON ASL WORD DATA — v2
  Fixed: learning rate, normalization, dropout
=====================================================
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── CONFIG ─────────────────────────────────────────
WORDS = [
    'hello', 'thanks', 'iloveyou',
    'hi', 'my', 'name', 'is',
    'what', 'how', 'you',
    'please', 'sorry', 'help',
    'yes', 'no', 'good', 'bad',
    'neutral'
]

DATA_PATH = 'MP_Data'
SEQUENCE_LENGTH = 30
MODEL_OUTPUT = 'action.h5'

# ─── LOAD & INSPECT DATA ───────────────────────────
print(f"\n{'='*50}")
print(f"  Loading & inspecting training data...")
print(f"{'='*50}\n")

label_map = {word: idx for idx, word in enumerate(WORDS)}
sequences, labels = [], []
quality_report = {}

for word in WORDS:
    word_path = os.path.join(DATA_PATH, word)
    if not os.path.exists(word_path):
        print(f'  [WARNING] No data for "{word}" — skipping!')
        quality_report[word] = {'loaded': 0, 'good': 0}
        continue

    seq_folders = sorted(
        [f for f in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, f))],
        key=lambda x: int(x)
    )
    loaded = 0
    good_frames_total = 0
    total_frames = 0

    for seq_folder in seq_folders:
        window = []
        complete = True
        good_frames = 0

        for frame_num in range(SEQUENCE_LENGTH):
            npy_file = os.path.join(word_path, seq_folder, f'{frame_num}.npy')
            if os.path.exists(npy_file):
                frame_data = np.load(npy_file)
                window.append(frame_data)
                # Check if hands were detected (not all zeros in hand region)
                # Pose=132, Face=1404, LH=63, RH=63 -> hands start at index 1536
                lh = frame_data[1536:1599]
                rh = frame_data[1599:1662]
                if np.any(lh != 0) or np.any(rh != 0):
                    good_frames += 1
                total_frames += 1
            else:
                complete = False
                break

        if complete and len(window) == SEQUENCE_LENGTH:
            sequences.append(window)
            labels.append(label_map[word])
            loaded += 1
            good_frames_total += good_frames

    avg_good = (good_frames_total / (loaded * SEQUENCE_LENGTH) * 100) if loaded > 0 else 0
    quality_report[word] = {'loaded': loaded, 'good': avg_good}
    status = "OK" if avg_good > 50 else "LOW"
    print(f'  "{word}" — {loaded} seqs | {avg_good:.0f}% frames with hands [{status}]')

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f'\n  Total samples: {X.shape[0]}')
print(f'  Input shape:   {X.shape[1:]}')
print(f'  Classes:       {len(WORDS)}')

# ─── NORMALIZE DATA ─────────────────────────────────
# Normalize each feature across the entire dataset
print(f'\n  Normalizing data...')
X_shape = X.shape
X_flat = X.reshape(-1, X_shape[-1])  # (samples * frames, features)

# Per-feature mean and std
mean = np.mean(X_flat, axis=0)
std = np.std(X_flat, axis=0)
std[std == 0] = 1  # Prevent division by zero

X_flat = (X_flat - mean) / std
X = X_flat.reshape(X_shape)

# Save normalization params for inference
np.save('norm_mean.npy', mean)
np.save('norm_std.npy', std)
print(f'  Saved normalization params (norm_mean.npy, norm_std.npy)')

# ─── SPLIT DATA ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f'  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ─── BUILD IMPROVED MODEL ──────────────────────────
print(f"\n{'='*50}")
print(f"  Building improved LSTM model...")
print(f"{'='*50}\n")

model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, 1662)))
model.add(LSTM(64, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(len(WORDS), activation='softmax'))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# ─── TRAIN ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Training (this may take a few minutes)...")
print(f"{'='*50}\n")

callbacks = [
    EarlyStopping(monitor='val_categorical_accuracy', patience=30, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=8,
    callbacks=callbacks,
    validation_data=(X_test, y_test),
    verbose=1
)

# ─── EVALUATE ───────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

print(f"\n{'='*50}")
print(f"  RESULTS")
print(f"{'='*50}")
print(f"  Train Accuracy: {train_acc * 100:.1f}%")
print(f"  Test Accuracy:  {accuracy * 100:.1f}%")
print(f"{'='*50}")

# ─── SAVE ───────────────────────────────────────────
model.save_weights(MODEL_OUTPUT)
print(f"\n  Model saved to: {MODEL_OUTPUT}")
print(f"  Words: {WORDS}")
print(f"\n  Update 'actions' in app.py/main.py to:")
print(f"  actions = np.array({WORDS})")
print(f"\n{'='*50}")
print(f"  Done! Restart app.py to use the new model.")
print(f"{'='*50}\n")
