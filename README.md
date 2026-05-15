# ASL Sign Language Recognition Platform

> **Real-time ASL sign language recognition using Dual-Brain AI architecture — LSTM for dynamic gesture words + Random Forest for static letters.**

---

## Features

- **55 Signs Detected** — 26 letters (A-Z), 10 digits (0-9), space, dot, and 17 gesture words
- **17 Dynamic Gesture Words** — hello, thanks, iloveyou, hi, my, name, is, what, how, you, please, sorry, help, yes, no, good, bad
- **Full Phrase Detection** — Sign "Hello my name is Haitam" in real-time (gestures + letter spelling)
- **4-Language Translation** — English, French, Spanish, Arabic + Text-to-Speech
- **Professional Web Platform** — Flask-based with learning mode, practice mode, and progress tracking
- **55 Interactive Lessons** — Coursera-style learning with camera validation

---

## Architecture

```
              ┌────────────────────────────────┐
              │     Camera (DroidCam/Webcam)    │
              └──────────────┬─────────────────┘
                             │
                     MediaPipe Holistic
                  (1662 keypoints extracted)
                      │              │
           ┌──────────┘              └──────────┐
           ▼                                    ▼
   ┌───────────────┐                  ┌───────────────┐
   │  Brain 1      │                  │  Brain 2      │
   │ Random Forest │                  │    LSTM       │
   │  (model.p)    │                  │ (action.h5)   │
   │               │                  │               │
   │ Letters A-Z   │                  │  17 gesture   │
   │ Digits 0-9    │                  │  words        │
   │ Space, Dot    │                  │               │
   │               │                  │               │
   │ Accuracy:     │                  │ Accuracy:     │
   │   100%        │                  │   98.7%       │
   └───────┬───────┘                  └───────┬───────┘
           └──────────────┬───────────────────┘
                          ▼
               Phrase Construction
                          │
               ┌──────────┼──────────┐
               ▼          ▼          ▼
          Translation   Text     Text-to-Speech
          (4 langs)   Display      (gTTS)
```

---

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Haitammos/Sign-Language-Platforme.git
cd Sign-Language-Platforme
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Run the Web Platform
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

### 3. Run the Demo (Guided)
```bash
python demo_hello_my_name_is.py
```
Follow the on-screen instructions. Press **Q** to confirm each sign.

---

## Project Structure

```
├── app.py                    # Flask web server (main platform)
├── main.py                   # Standalone tkinter version
├── demo_hello_my_name_is.py  # Guided phrase demo
├── diagnostic.py             # Real-time detection diagnostic tool
│
├── model.p                   # Random Forest model (letters)
├── action.h5                 # LSTM model (gesture words)
├── norm_mean.npy             # LSTM normalization params
├── norm_std.npy              # LSTM normalization params
│
├── collect_words.py          # Collect LSTM training data
├── collect_neutral.py        # Collect neutral class data
├── train_lstm.py             # Train LSTM model
├── collectImgs.py            # Collect RF training images
├── createDataset.py          # Create RF dataset
├── trainClassifier.py        # Train RF model
│
├── templates/                # Web UI templates
│   ├── base.html
│   ├── index.html            # Landing page
│   ├── learn.html            # Learning grid
│   ├── lesson.html           # Individual lesson
│   └── practice.html         # Free practice mode
│
├── static/                   # CSS, JS, gesture images
│   ├── style.css
│   ├── script.js
│   └── gesture_*.png
│
└── requirements.txt
```

---

## Technologies

| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| TensorFlow / Keras | LSTM model for dynamic gestures |
| Scikit-Learn | Random Forest model for static signs |
| MediaPipe Holistic | Real-time hand/body/face tracking |
| OpenCV | Camera capture & video processing |
| Flask | Web server |
| gTTS + Pygame | Text-to-speech in 4 languages |
| Deep Translator | Multi-language translation |

---

## How to Retrain Models

### Retrain LSTM (gesture words)
```bash
python collect_words.py      # Record 30 sequences per word
python collect_neutral.py    # Record neutral class (sit still)
python train_lstm.py         # Train and save action.h5
```

### Retrain Random Forest (letters)
```bash
python collectImgs.py        # Capture hand images
python createDataset.py      # Extract features
python trainClassifier.py    # Train and save model.p
```

---

## Requirements

```
flask
mediapipe
tensorflow
opencv-python
scikit-learn
numpy
gtts
pygame
deep-translator
```

---

## License

MIT License
