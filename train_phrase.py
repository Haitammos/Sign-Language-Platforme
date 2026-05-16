"""
=============================================================================
  TRAIN PHRASE MODEL v2 — HANDS ONLY (126 features)
  Removes face/pose noise, focuses only on hand landmarks
=============================================================================
"""
import os, numpy as np, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

WORDS = ['hello', 'my', 'name', 'is', 'neutral']
DATA_PATH = 'MP_Data_Phrase'
SEQUENCE_LENGTH = 30
# In 1662 features: pose=132, face=1404, lh=63, rh=63
# We only keep hands: indices 1536:1662 (126 features)
HAND_START = 1536
HAND_END = 1662
HAND_FEATURES = HAND_END - HAND_START  # 126

print(f"\n{'='*55}")
print(f"  Loading HANDS-ONLY data ({HAND_FEATURES} features)")
print(f"{'='*55}\n")

label_map = {word: idx for idx, word in enumerate(WORDS)}
sequences, labels = [], []

for word in WORDS:
    word_path = os.path.join(DATA_PATH, word)
    if not os.path.exists(word_path):
        print(f'  [WARN] No data for "{word}"')
        continue

    seq_folders = sorted(
        [f for f in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, f))],
        key=lambda x: int(x)
    )
    loaded = 0
    for seq_folder in seq_folders:
        window = []
        complete = True
        for frame_num in range(SEQUENCE_LENGTH):
            npy_file = os.path.join(word_path, seq_folder, f'{frame_num}.npy')
            if os.path.exists(npy_file):
                full = np.load(npy_file)
                hands_only = full[HAND_START:HAND_END]  # 126 features
                window.append(hands_only)
            else:
                complete = False
                break
        if complete and len(window) == SEQUENCE_LENGTH:
            sequences.append(window)
            labels.append(label_map[word])
            loaded += 1

    print(f'  "{word}" — {loaded} sequences')

X = np.array(sequences)
y = to_categorical(labels).astype(int)
print(f'\n  Samples: {X.shape[0]} | Shape: {X.shape[1:]} | Classes: {len(WORDS)}')

# Normalize
print(f'  Normalizing...')
X_flat = X.reshape(-1, HAND_FEATURES)
mean = np.mean(X_flat, axis=0)
std = np.std(X_flat, axis=0)
std[std == 0] = 1
X_flat = (X_flat - mean) / std
X = X_flat.reshape(X.shape[0], SEQUENCE_LENGTH, HAND_FEATURES)

np.save('phrase_norm_mean.npy', mean)
np.save('phrase_norm_std.npy', std)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f'  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# Model — smaller, faster, HANDS ONLY input
print(f"\n{'='*55}")
print(f"  Building model (input: {HAND_FEATURES} features)...")
print(f"{'='*55}\n")

model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, HAND_FEATURES)))
model.add(LSTM(64, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(len(WORDS), activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)
model.summary()

# Train
print(f"\n  Training...\n")
callbacks = [
    EarlyStopping(monitor='val_categorical_accuracy', patience=50,
                  restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15,
                      min_lr=1e-6, verbose=1)
]

model.fit(
    X_train, y_train,
    epochs=500, batch_size=8,
    callbacks=callbacks,
    validation_data=(X_test, y_test),
    verbose=1
)

# Results
loss, acc = model.evaluate(X_test, y_test, verbose=0)
t_loss, t_acc = model.evaluate(X_train, y_train, verbose=0)

print(f"\n{'='*55}")
print(f"  Train: {t_acc*100:.1f}% | Test: {acc*100:.1f}%")
print(f"{'='*55}")

model.save_weights('phrase_model.h5')
print(f"  Saved: phrase_model.h5")
print(f"  Features: HANDS ONLY ({HAND_FEATURES})")
print(f"  Now run: python demo_hello_my_name_is.py")
print(f"{'='*55}\n")
