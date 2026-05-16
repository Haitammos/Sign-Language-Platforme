"""
  TRAIN GESTURE RF — Uses existing data from MP_Data_Phrase
"""
import os, numpy as np, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = 'MP_Data_Phrase'
WORDS = ['hello', 'my', 'name', 'is', 'how', 'you', 'neutral']
SEQ_LEN = 30
# In 1662-feature vectors: hands start at index 1536
# lh = [1536:1599] (63), rh = [1599:1662] (63)
HAND_START = 1536
HAND_END = 1662

print(f"\n{'='*55}")
print(f"  Training Gesture RF from MP_Data_Phrase")
print(f"{'='*55}\n")

X, y = [], []
for word in WORDS:
    wp = os.path.join(DATA_PATH, word)
    if not os.path.exists(wp): print(f'  [WARN] No {word}'); continue
    sfs = sorted([f for f in os.listdir(wp) if os.path.isdir(os.path.join(wp,f))], key=lambda x: int(x))
    loaded = 0
    for sf in sfs:
        frames = []
        ok = True
        for fn in range(SEQ_LEN):
            fp = os.path.join(wp, sf, f'{fn}.npy')
            if os.path.exists(fp):
                full = np.load(fp)
                hands = full[HAND_START:HAND_END]  # 126 features (lh+rh)
                frames.append(hands)
            else:
                ok = False; break
        if ok and len(frames) == SEQ_LEN:
            seq = np.array(frames)  # (30, 126)
            feat_mean = np.mean(seq, axis=0)
            feat_std = np.std(seq, axis=0)
            feat_max = np.max(seq, axis=0)
            feat_min = np.min(seq, axis=0)
            diffs = np.diff(seq, axis=0)
            feat_vel_mean = np.mean(np.abs(diffs), axis=0)
            feat_vel_std = np.std(diffs, axis=0)
            features = np.concatenate([feat_mean, feat_std, feat_max, feat_min, feat_vel_mean, feat_vel_std])
            X.append(features)
            y.append(word)
            loaded += 1
    print(f'  "{word}": {loaded} sequences -> {len(features)} features')

X = np.array(X); y = np.array(y)
print(f'\n  Total: {len(X)} samples, {X.shape[1]} features')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'  Train: {len(X_train)} | Test: {len(X_test)}')

model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=3, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"\n{'='*55}")
print(f"  Test Accuracy:  {acc*100:.1f}%")
print(f"  CV Accuracy:    {cv.mean()*100:.1f}% (+/- {cv.std()*100:.1f}%)")
print(f"{'='*55}")
print(f"\n{classification_report(y_test, y_pred)}")

with open('gesture_rf.p', 'wb') as f:
    pickle.dump({'model': model, 'words': WORDS, 'hand_start': HAND_START, 'hand_end': HAND_END}, f)
print(f"  Saved: gesture_rf.p")
print(f"  Run: python demo_final.py")
