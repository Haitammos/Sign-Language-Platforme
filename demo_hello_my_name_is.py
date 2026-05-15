"""
=============================================================================
  GUIDED DEMO: "Hello my name is Haitam"
  
  OPTIMIZED VERSION — Fast, no lag
  
  HOW IT WORKS:
  1. Screen tells you what to sign
  2. You perform the sign — AI shows what it detects
  3. Press Q when done → moves to next sign
  4. Press R to restart | ESC to quit
=============================================================================
"""
import os, sys, cv2, time, pickle, numpy as np, mediapipe as mp

# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
# Force CPU but optimize it
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization

# =============================================================================
# 1. LOAD MODELS
# =============================================================================
print("[*] Loading models...")

actions = np.array([
    'hello', 'thanks', 'iloveyou', 'hi', 'my', 'name', 'is',
    'what', 'how', 'you', 'please', 'sorry', 'help',
    'yes', 'no', 'good', 'bad', 'neutral'
])

lstm_model = Sequential()
lstm_model.add(Input(shape=(30, 1662)))
lstm_model.add(LSTM(64, return_sequences=True, activation='tanh'))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(128, return_sequences=True, activation='tanh'))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(64, return_sequences=False, activation='tanh'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(BatchNormalization())
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(len(actions), activation='softmax'))

try:
    lstm_model.load_weights('action.h5')
    print("[OK] LSTM model loaded")
except Exception as e:
    print(f"[ERROR] action.h5: {e}")
    sys.exit(1)

# Normalization
try:
    norm_mean = np.load('norm_mean.npy')
    norm_std = np.load('norm_std.npy')
    norm_std[norm_std == 0] = 1
    print("[OK] Normalization loaded")
except:
    norm_mean = None
    norm_std = None

# Random Forest
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

try:
    with open('./model.p', 'rb') as f:
        model_dict = RenameUnpickler(f).load()
    rf_model = model_dict['model']
    print("[OK] Random Forest loaded")
except Exception as e:
    print(f"[ERROR] model.p: {e}")
    sys.exit(1)

LABELS = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
    19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
    36:' ',37:'.'
}

# Warm up LSTM (first call is always slow)
print("[*] Warming up LSTM...")
dummy = np.zeros((1, 30, 1662), dtype=np.float32)
lstm_model.predict(dummy, verbose=0)
print("[OK] Ready!\n")

# =============================================================================
# 2. HELPERS
# =============================================================================

def extract_keypoints(results):
    pose = np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[r.x,r.y,r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def get_letter(hand_lm):
    data_aux, x_, y_ = [], [], []
    for lm in hand_lm.landmark:
        x_.append(lm.x); y_.append(lm.y)
    for lm in hand_lm.landmark:
        data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
    if len(data_aux) < 42: data_aux.extend([0]*(42-len(data_aux)))
    elif len(data_aux) > 42: data_aux = data_aux[:42]
    pred = rf_model.predict([np.asarray(data_aux)])
    try: return LABELS[int(pred[0])]
    except: return None

# =============================================================================
# 3. DEMO STEPS
# =============================================================================

STEPS = [
    {'text': 'Sign: "HELLO"',   'type': 'gesture', 'show': 'Hello'},
    {'text': 'Sign: "MY"',      'type': 'gesture', 'show': 'my'},
    {'text': 'Sign: "NAME"',    'type': 'gesture', 'show': 'name'},
    {'text': 'Sign: "IS"',      'type': 'gesture', 'show': 'is'},
    {'text': 'Show letter: H',  'type': 'letter',  'show': 'H'},
    {'text': 'Show letter: A',  'type': 'letter',  'show': 'a'},
    {'text': 'Show letter: I',  'type': 'letter',  'show': 'i'},
    {'text': 'Show letter: T',  'type': 'letter',  'show': 't'},
    {'text': 'Show letter: A',  'type': 'letter',  'show': 'a'},
    {'text': 'Show letter: M',  'type': 'letter',  'show': 'm'},
]

# =============================================================================
# 4. MAIN LOOP
# =============================================================================

print("="*55)
print("  DEMO: 'Hello my name is Haitam'")
print("  Q = confirm sign | R = restart | ESC = quit")
print("="*55 + "\n")

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] No camera!"); sys.exit(1)

# Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

step_idx = 0
parts = []
seq_buf = []
predict_every = 3   # Only run LSTM every N frames (huge speedup)
frame_count = 0
live_det = ''
live_conf = 0.0
live_type = ''
fps_time = time.time()
fps = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    model_complexity=0         # 0 = fastest, 1 = balanced, 2 = best
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame_count += 1
        h, w = frame.shape[:2]

        # --- FPS counter ---
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # --- MediaPipe ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        has_hands = (results.left_hand_landmarks is not None or
                     results.right_hand_landmarks is not None)

        # Draw hand landmarks
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                   mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                   mp_holistic.HAND_CONNECTIONS)

        # --- AI Detection (only every N frames for speed) ---
        if has_hands and step_idx < len(STEPS) and frame_count % predict_every == 0:
            step = STEPS[step_idx]

            if step['type'] == 'gesture':
                kp = extract_keypoints(results)
                seq_buf.append(kp)
                if len(seq_buf) > 30: seq_buf = seq_buf[-30:]

                if len(seq_buf) == 30:
                    inp = np.array(seq_buf, dtype=np.float32)
                    if norm_mean is not None:
                        inp = (inp - norm_mean) / norm_std
                    res = lstm_model.predict(
                        np.expand_dims(inp, axis=0), verbose=0
                    )[0]
                    idx = np.argmax(res)
                    word = actions[idx]
                    if word != 'neutral' and res[idx] > 0.40:
                        live_det = word.capitalize()
                        live_conf = float(res[idx])
                        live_type = 'LSTM'
                    else:
                        live_det = '...'
                        live_conf = 0.0

            elif step['type'] == 'letter':
                hl = results.right_hand_landmarks or results.left_hand_landmarks
                if hl:
                    letter = get_letter(hl)
                    if letter:
                        live_det = letter
                        live_conf = 1.0
                        live_type = 'RF'
        elif not has_hands:
            seq_buf.clear()
            live_det = ''
            live_conf = 0.0

        # =================================================================
        # DRAW UI
        # =================================================================

        # --- Top bar ---
        cv2.rectangle(frame, (0, 0), (w, 85), (10, 10, 10), -1)

        if step_idx < len(STEPS):
            step = STEPS[step_idx]

            # Step counter + FPS
            cv2.putText(frame, f"Step {step_idx+1}/{len(STEPS)}   FPS: {fps}",
                        (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

            # Main instruction
            cv2.putText(frame, step['text'], (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)

            # Help
            cv2.putText(frame, "Q = confirm  |  R = restart  |  ESC = quit",
                        (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

        # Hands indicator
        color = (0,255,0) if has_hands else (0,0,255)
        label = "HANDS" if has_hands else "NO HANDS"
        cv2.putText(frame, label, (w-120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Live detection box ---
        if live_det and live_det != '...':
            bx = w - 220
            cv2.rectangle(frame, (bx, 95), (w-10, 175), (20,20,20), -1)
            cv2.rectangle(frame, (bx, 95), (w-10, 175), (60,60,60), 1)

            cv2.putText(frame, f"{live_type}:", (bx+10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,140), 1)

            # Detection name — big
            det_color = ((0,255,0) if live_conf > 0.70
                         else (0,180,255) if live_conf > 0.50
                         else (100,100,255))
            cv2.putText(frame, live_det, (bx+10, 148),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, det_color, 2)

            if live_type == 'LSTM':
                # Confidence bar
                bar_w = int(180 * live_conf)
                cv2.rectangle(frame, (bx+10, 158), (bx+190, 168), (40,40,40), -1)
                cv2.rectangle(frame, (bx+10, 158), (bx+10+bar_w, 168), det_color, -1)
                cv2.putText(frame, f"{live_conf:.0%}", (bx+140, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, det_color, 1)

        # --- Bottom: phrase panel ---
        py = h - 75
        cv2.rectangle(frame, (0, py), (w, h), (10,10,10), -1)
        cv2.line(frame, (0, py), (w, py), (50,50,50), 1)

        if parts:
            # Build display string
            display = ''
            spell = ''
            for p in parts:
                if len(p) == 1:
                    spell += p
                else:
                    if spell:
                        display += ' ' + spell
                        spell = ''
                    display += (' ' if display else '') + p
            if spell:
                display += ' ' + spell

            cv2.putText(frame, display.strip(), (15, py+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Progress bar
            prog = len(parts) / len(STEPS)
            bw = int((w-30) * prog)
            cv2.rectangle(frame, (15, py+50), (w-15, py+60), (40,40,40), -1)
            cv2.rectangle(frame, (15, py+50), (15+bw, py+60), (0,200,100), -1)
        else:
            cv2.putText(frame, "Sign your first gesture...", (15, py+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 1)

        # --- COMPLETE overlay ---
        if step_idx >= len(STEPS):
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (w,h), (0,60,0), -1)
            frame = cv2.addWeighted(ov, 0.3, frame, 0.7, 0)

            cv2.rectangle(frame, (0,0), (w,85), (0,80,0), -1)
            cv2.putText(frame, "PHRASE COMPLETE!", (w//2-200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,100), 3)

            # Full phrase
            full = ''
            sp = ''
            for p in parts:
                if len(p)==1: sp += p
                else:
                    if sp: full += ' ' + sp; sp = ''
                    full += (' ' if full else '') + p
            if sp: full += ' ' + sp

            cv2.putText(frame, full.strip(), (w//2-200, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, "R = restart | ESC = quit", (w//2-140, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # --- Show ---
        cv2.imshow('ASL Demo - Hello my name is Haitam', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') and step_idx < len(STEPS):
            parts.append(STEPS[step_idx]['show'])
            print(f"  [Step {step_idx+1}] '{STEPS[step_idx]['show']}'")
            step_idx += 1
            seq_buf.clear()
            live_det = ''
            live_conf = 0.0

            if step_idx >= len(STEPS):
                full = ''
                sp = ''
                for p in parts:
                    if len(p)==1: sp += p
                    else:
                        if sp: full += ' ' + sp; sp = ''
                        full += (' ' if full else '') + p
                if sp: full += ' ' + sp

                print(f"\n{'='*55}")
                print(f"  COMPLETE: {full.strip()}")
                print(f"{'='*55}\n")

                try:
                    from gtts import gTTS
                    import pygame
                    pygame.mixer.init()
                    tts = gTTS(text=full.strip(), lang='en')
                    tts.save('_demo_tts.mp3')
                    pygame.mixer.music.load('_demo_tts.mp3')
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    os.remove('_demo_tts.mp3')
                except Exception as e:
                    print(f"  [TTS skipped: {e}]")

        elif key == ord('r'):
            step_idx = 0
            parts = []
            seq_buf = []
            live_det = ''
            live_conf = 0.0
            print("\n  >>> RESTART <<<\n")

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("\n  Demo ended.\n")
