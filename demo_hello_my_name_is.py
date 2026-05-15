"""
=============================================================================
  GUIDED DEMO: "Hello my name is Haitam"
  
  HOW IT WORKS:
  1. Screen tells you what to sign
  2. You perform the sign — AI shows what it detects in real-time
  3. Press Q when you're done → moves to next sign
  4. Repeat until the full phrase is built
=============================================================================
"""
import os, cv2, time, pickle, numpy as np, mediapipe as mp, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization

# =============================================================================
# 1. LOAD MODELS
# =============================================================================

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
    exit(1)

try:
    norm_mean = np.load('norm_mean.npy')
    norm_std = np.load('norm_std.npy')
    norm_std[norm_std == 0] = 1
    print("[OK] Normalization loaded")
except:
    norm_mean = None
    norm_std = None

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

try:
    with open('./model.p', 'rb') as f:
        model_dict = RenameUnpickler(f).load()
    rf_model = model_dict['model']
    print("[OK] Random Forest model loaded")
except Exception as e:
    print(f"[ERROR] model.p: {e}")
    exit(1)

LABELS_DICT = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
    19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
    36:' ',37:'.'
}

# =============================================================================
# 2. HELPERS
# =============================================================================

def extract_keypoints(results):
    pose = np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[r.x,r.y,r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def get_rf_pred(hand_lm):
    data_aux, x_, y_ = [], [], []
    for lm in hand_lm.landmark:
        x_.append(lm.x); y_.append(lm.y)
    for lm in hand_lm.landmark:
        data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
    if len(data_aux) < 42: data_aux.extend([0]*(42-len(data_aux)))
    elif len(data_aux) > 42: data_aux = data_aux[:42]
    pred = rf_model.predict([np.asarray(data_aux)])
    try: return LABELS_DICT[int(pred[0])]
    except: return None

# =============================================================================
# 3. DEMO SEQUENCE
# =============================================================================

# Each step: (what to display, type, value)
STEPS = [
    # LSTM gesture words
    {'instruction': 'Sign: "HELLO"',   'type': 'gesture', 'display': 'Hello'},
    {'instruction': 'Sign: "MY"',      'type': 'gesture', 'display': 'my'},
    {'instruction': 'Sign: "NAME"',    'type': 'gesture', 'display': 'name'},
    {'instruction': 'Sign: "IS"',      'type': 'gesture', 'display': 'is'},
    # RF letters for spelling
    {'instruction': 'Show letter: H',  'type': 'letter',  'display': 'H'},
    {'instruction': 'Show letter: A',  'type': 'letter',  'display': 'a'},
    {'instruction': 'Show letter: I',  'type': 'letter',  'display': 'i'},
    {'instruction': 'Show letter: T',  'type': 'letter',  'display': 't'},
    {'instruction': 'Show letter: A',  'type': 'letter',  'display': 'a'},
    {'instruction': 'Show letter: M',  'type': 'letter',  'display': 'm'},
]

current_step = 0
phrase_parts = []
sequence_buffer = []

print(f"\n{'='*60}")
print(f"  GUIDED DEMO: 'Hello my name is Haitam'")
print(f"  Perform each sign, then press Q to confirm")
print(f"  Press ESC to quit")
print(f"{'='*60}\n")

# =============================================================================
# 4. MAIN LOOP
# =============================================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No camera!"); exit(1)

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        has_hands = results.left_hand_landmarks or results.right_hand_landmarks

        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- Real-time AI detection (display only) ---
        live_detection = ''
        live_confidence = 0.0
        live_type = ''

        if has_hands and current_step < len(STEPS):
            step = STEPS[current_step]

            if step['type'] == 'gesture':
                # LSTM prediction
                kp = extract_keypoints(results)
                sequence_buffer.append(kp)
                if len(sequence_buffer) > 30:
                    sequence_buffer = sequence_buffer[-30:]
                if len(sequence_buffer) == 30:
                    inp = np.array(sequence_buffer)
                    if norm_mean is not None:
                        inp = (inp - norm_mean) / norm_std
                    res = lstm_model.predict(np.expand_dims(inp, axis=0), verbose=0)[0]
                    idx = np.argmax(res)
                    word = actions[idx]
                    if word != 'neutral':
                        live_detection = word.capitalize()
                        live_confidence = res[idx]
                        live_type = 'LSTM'

            elif step['type'] == 'letter':
                # RF prediction
                hl = results.right_hand_landmarks or results.left_hand_landmarks
                if hl:
                    letter = get_rf_pred(hl)
                    if letter:
                        live_detection = letter
                        live_confidence = 1.0
                        live_type = 'RF'
        else:
            sequence_buffer.clear()

        # =================================================================
        # DRAW UI
        # =================================================================

        # --- Top bar: dark background ---
        cv2.rectangle(frame, (0, 0), (w, 90), (10, 10, 10), -1)

        if current_step < len(STEPS):
            step = STEPS[current_step]

            # Step counter
            step_text = f"Step {current_step + 1}/{len(STEPS)}"
            cv2.putText(frame, step_text, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            # Main instruction (big, yellow)
            cv2.putText(frame, step['instruction'], (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)

            # Help text
            cv2.putText(frame, "Press Q when done | ESC to quit", (15, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # --- Hands status ---
        if has_hands:
            cv2.putText(frame, "HANDS OK", (w - 130, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO HANDS", (w - 130, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # --- Live detection display (right side) ---
        if live_detection:
            conf_color = ((0, 255, 0) if live_confidence > 0.70
                          else (0, 180, 255) if live_confidence > 0.50
                          else (80, 80, 255))

            # Detection box
            box_x = w - 250
            cv2.rectangle(frame, (box_x, 100), (w - 10, 185), (20, 20, 20), -1)
            cv2.rectangle(frame, (box_x, 100), (w - 10, 185), (60, 60, 60), 1)

            cv2.putText(frame, f"{live_type} detects:", (box_x + 10, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.putText(frame, live_detection, (box_x + 10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)
            if live_type == 'LSTM':
                cv2.putText(frame, f"{live_confidence:.0%}", (box_x + 10, 178),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)

        # --- Bottom panel: phrase being built ---
        panel_y = h - 80
        cv2.rectangle(frame, (0, panel_y), (w, h), (10, 10, 10), -1)
        cv2.line(frame, (0, panel_y), (w, panel_y), (50, 50, 50), 1)

        # Build display phrase
        if phrase_parts:
            display_phrase = ''
            spelling = ''
            for p in phrase_parts:
                if len(p) == 1:
                    spelling += p
                else:
                    if spelling:
                        display_phrase += ' ' + spelling
                        spelling = ''
                    display_phrase += (' ' if display_phrase else '') + p
            if spelling:
                display_phrase += ' ' + spelling

            cv2.putText(frame, display_phrase.strip(), (15, panel_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Progress bar
            progress = len(phrase_parts) / len(STEPS)
            bar_w = int((w - 30) * progress)
            cv2.rectangle(frame, (15, panel_y + 55), (w - 15, panel_y + 65), (40, 40, 40), -1)
            cv2.rectangle(frame, (15, panel_y + 55), (15 + bar_w, panel_y + 65), (0, 200, 100), -1)
        else:
            cv2.putText(frame, "Waiting for first sign...", (15, panel_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)

        # --- PHRASE COMPLETE overlay ---
        if current_step >= len(STEPS):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 60, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            cv2.rectangle(frame, (0, 0), (w, 90), (0, 80, 0), -1)
            cv2.putText(frame, "PHRASE COMPLETE!", (w//2 - 190, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

            # Show full phrase
            full = ''
            spelling = ''
            for p in phrase_parts:
                if len(p) == 1:
                    spelling += p
                else:
                    if spelling:
                        full += ' ' + spelling
                        spelling = ''
                    full += (' ' if full else '') + p
            if spelling:
                full += ' ' + spelling

            cv2.putText(frame, full.strip(), (w//2 - 200, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, "Press R to restart | ESC to quit", (w//2 - 180, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # --- Show ---
        cv2.imshow('ASL Demo - Hello my name is Haitam', frame)

        key = cv2.waitKey(10) & 0xFF

        # Q = confirm current step and advance
        if key == ord('q') and current_step < len(STEPS):
            step = STEPS[current_step]
            phrase_parts.append(step['display'])
            print(f"  [STEP {current_step+1}] Confirmed: '{step['display']}'")
            current_step += 1
            sequence_buffer.clear()

            if current_step >= len(STEPS):
                print(f"\n{'='*60}")
                full_phrase = ''
                sp = ''
                for p in phrase_parts:
                    if len(p) == 1:
                        sp += p
                    else:
                        if sp:
                            full_phrase += ' ' + sp
                            sp = ''
                        full_phrase += (' ' if full_phrase else '') + p
                if sp:
                    full_phrase += ' ' + sp
                print(f"  COMPLETE: {full_phrase.strip()}")
                print(f"{'='*60}\n")

                # Text-to-speech
                try:
                    from gtts import gTTS
                    import pygame
                    pygame.mixer.init()
                    tts = gTTS(text=full_phrase.strip(), lang='en')
                    tts.save('_demo_speech.mp3')
                    pygame.mixer.music.load('_demo_speech.mp3')
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    os.remove('_demo_speech.mp3')
                except:
                    pass

        # R = restart
        elif key == ord('r'):
            current_step = 0
            phrase_parts = []
            sequence_buffer = []
            print("\n  >>> RESTARTED <<<\n")

        # ESC = quit
        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("\n  Demo ended.\n")
