# ASL Sign Language Recognition Platform

> Real-time ASL sign language recognition using **Dual-Brain AI** â€” LSTM for dynamic gesture words + Random Forest for static letters.

---

## Demo: "Hello my name is Omar"

The platform detects the phrase **"Hello my name is Omar"** automatically:
- **Hello, my, name, is** â†’ Dynamic gestures detected by LSTM (hands-only, 126 features)
- **H, a, i, t, a, m** â†’ Static letters detected by Random Forest

---

## Quick Start (Windows)

### Option 1: Automatic Setup
```bat
setup.bat
```

### Option 2: Manual Setup
```bash
git clone https://github.com/Omarmos/Sign-Language-Platforme.git
cd Sign-Language-Platforme
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Step 1: Collect Training Data (~5 min)
```bash
python collect_phrase.py
```
- Records 30 sequences Ã— 5 words (hello, my, name, is, neutral)
- Press **Q** to record each sequence, **S** to skip a word

### Step 2: Train the Model (~2 min)
```bash
python train_phrase.py
```
- Trains hands-only LSTM (126 features, 5 classes)
- Saves: `phrase_model.h5`, `phrase_norm_mean.npy`, `phrase_norm_std.npy`

### Step 3: Run the Demo
```bash
# Automatic detection (no buttons needed):
python demo_perfect.py

# Guided demo (press Q to confirm each sign):
python demo_hello_my_name_is.py
```

### Step 4: Run the Web Platform
```bash
python app.py
# Open http://localhost:5000
```

---

## Architecture

```
              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
              â”‚   Camera (Webcam)    â”‚
              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                         â”‚
                 MediaPipe Holistic
                         â”‚
              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
              â”‚                      â”‚
      â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
      â”‚   Brain 1      â”‚    â”‚   Brain 2      â”‚
      â”‚ Random Forest  â”‚    â”‚ LSTM (hands)   â”‚
      â”‚  (model.p)     â”‚    â”‚(phrase_model)  â”‚
      â”‚                â”‚    â”‚                â”‚
      â”‚ Letters A-Z    â”‚    â”‚ hello, my      â”‚
      â”‚ Digits 0-9     â”‚    â”‚ name, is       â”‚
      â”‚ Space, Dot     â”‚    â”‚ neutral        â”‚
      â”‚                â”‚    â”‚                â”‚
      â”‚ Accuracy: 100% â”‚    â”‚ Accuracy: 100% â”‚
      â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                         â”‚
              Phrase: "Hello my name is Omar"
                         â”‚
              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
              â”‚   Text-to-Speech     â”‚
              â”‚   4-lang Translation â”‚
              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Project Files

| File | Purpose |
|---|---|
| `demo_perfect.py` | **Automatic** phrase detection demo |
| `demo_hello_my_name_is.py` | **Guided** demo (Q to confirm) |
| `app.py` | Flask web platform |
| `collect_phrase.py` | Collect gesture data (5 classes) |
| `train_phrase.py` | Train hands-only LSTM |
| `phrase_model.h5` | Trained LSTM weights |
| `model.p` | Random Forest model (letters) |
| `setup.bat` | Automatic setup script |

---

## Technologies

- Python 3.10 | TensorFlow | Scikit-Learn
- MediaPipe Holistic | OpenCV
- Flask | gTTS | Deep Translator

---

## License

MIT License

