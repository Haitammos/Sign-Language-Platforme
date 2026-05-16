import os
import time
import queue
import threading
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
from flask import Flask, render_template, Response, jsonify, request, send_file
from gtts import gTTS
import pygame
from deep_translator import GoogleTranslator

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

app = Flask(__name__)

# ============================================================
# 1. LOAD AI MODELS
# ============================================================

# --- LSTM Action Recognition Model (18 Words + neutral) ---
actions = np.array(['hello', 'thanks', 'iloveyou', 'hi', 'my', 'name', 'is',
                    'what', 'how', 'you', 'please', 'sorry', 'help',
                    'yes', 'no', 'good', 'bad', 'neutral'])

from tensorflow.keras.layers import Dropout, BatchNormalization

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
lstm_model.add(Dense(actions.shape[0], activation='softmax'))

try:
    lstm_model.load_weights('action.h5')
    print("LSTM Action Model loaded successfully!")
except Exception as e:
    print("Warning: Could not load action.h5:", e)

# Load normalization parameters for inference
try:
    norm_mean = np.load('norm_mean.npy')
    norm_std = np.load('norm_std.npy')
    norm_std[norm_std == 0] = 1
    print("Normalization params loaded.")
except:
    norm_mean = None
    norm_std = None
    print("Warning: norm_mean/norm_std not found — running without normalization.")

# --- Random Forest Alphabet Model (Letters) ---
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

with open('./model.p', 'rb') as f:
    model_dict = RenameUnpickler(f).load()
rf_model = model_dict['model']

# --- Gesture Random Forest (Dynamic gestures for phrase demo) ---
try:
    with open('gesture_rf.p', 'rb') as f:
        gesture_rf_data = pickle.load(f)
    gesture_rf_model = gesture_rf_data['model']
    GESTURE_WORDS = gesture_rf_data['words']
    print(f"Gesture RF loaded: {GESTURE_WORDS}")
except Exception as e:
    gesture_rf_model = None
    GESTURE_WORDS = []
    print(f"Warning: gesture_rf.p not loaded: {e}")

# ============================================================
# 2. CONFIGURATION
# ============================================================

LABELS_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5',
    32: '6', 33: '7', 34: '8', 35: '9', 36: ' ', 37: '.'
}
EXPECTED_FEATURES = 42

# Reverse lookup: letter -> class id
LETTER_TO_ID = {v: k for k, v in LABELS_DICT.items()}

# All learnable items
ALL_LESSONS = []
for i in range(26):
    if i not in LABELS_DICT: continue
    ALL_LESSONS.append({'id': str(i), 'label': LABELS_DICT[i], 'type': 'letter', 'display': LABELS_DICT[i]})
for i in range(26, 36):
    ALL_LESSONS.append({'id': str(i), 'label': LABELS_DICT[i], 'type': 'number', 'display': LABELS_DICT[i]})
ALL_LESSONS.append({'id': '36', 'label': ' ', 'type': 'special', 'display': 'SPACE'})
ALL_LESSONS.append({'id': '37', 'label': '.', 'type': 'special', 'display': 'DOT'})
ALL_LESSONS.append({'id': 'hello', 'label': 'Hello', 'type': 'gesture', 'display': 'Hello'})
ALL_LESSONS.append({'id': 'thanks', 'label': 'Thanks', 'type': 'gesture', 'display': 'Thanks'})
ALL_LESSONS.append({'id': 'iloveyou', 'label': 'Iloveyou', 'type': 'gesture', 'display': 'I Love You'})
ALL_LESSONS.append({'id': 'hi', 'label': 'Hi', 'type': 'gesture', 'display': 'Hi'})
ALL_LESSONS.append({'id': 'my', 'label': 'My', 'type': 'gesture', 'display': 'My'})
ALL_LESSONS.append({'id': 'name', 'label': 'Name', 'type': 'gesture', 'display': 'Name'})
ALL_LESSONS.append({'id': 'is', 'label': 'Is', 'type': 'gesture', 'display': 'Is'})
ALL_LESSONS.append({'id': 'what', 'label': 'What', 'type': 'gesture', 'display': 'What'})
ALL_LESSONS.append({'id': 'how', 'label': 'How', 'type': 'gesture', 'display': 'How'})
ALL_LESSONS.append({'id': 'you', 'label': 'You', 'type': 'gesture', 'display': 'You'})
ALL_LESSONS.append({'id': 'please', 'label': 'Please', 'type': 'gesture', 'display': 'Please'})
ALL_LESSONS.append({'id': 'sorry', 'label': 'Sorry', 'type': 'gesture', 'display': 'Sorry'})
ALL_LESSONS.append({'id': 'help', 'label': 'Help', 'type': 'gesture', 'display': 'Help'})
ALL_LESSONS.append({'id': 'yes', 'label': 'Yes', 'type': 'gesture', 'display': 'Yes'})
ALL_LESSONS.append({'id': 'no', 'label': 'No', 'type': 'gesture', 'display': 'No'})
ALL_LESSONS.append({'id': 'good', 'label': 'Good', 'type': 'gesture', 'display': 'Good'})
ALL_LESSONS.append({'id': 'bad', 'label': 'Bad', 'type': 'gesture', 'display': 'Bad'})

LANG_CODES = {'English': 'en', 'French': 'fr', 'Spanish': 'es', 'Arabic': 'ar'}

GESTURE_TRANSLATIONS = {
    'Hello':    {'English': 'Hello',      'French': 'Bonjour',     'Spanish': 'Hola',        'Arabic': 'مرحبا'},
    'Thanks':   {'English': 'Thanks',     'French': 'Merci',       'Spanish': 'Gracias',     'Arabic': 'شكرا'},
    'Iloveyou': {'English': 'I love you', 'French': "Je t'aime",   'Spanish': 'Te quiero',   'Arabic': 'أنا أحبك'},
    'Hi':       {'English': 'Hi',         'French': 'Salut',       'Spanish': 'Hola',        'Arabic': 'مرحبا'},
    'My':       {'English': 'My',         'French': 'Mon',         'Spanish': 'Mi',          'Arabic': 'لي'},
    'Name':     {'English': 'Name',       'French': 'Nom',         'Spanish': 'Nombre',      'Arabic': 'اسم'},
    'Is':       {'English': 'Is',         'French': 'Est',         'Spanish': 'Es',          'Arabic': 'هو'},
    'What':     {'English': 'What',       'French': 'Quoi',        'Spanish': 'Qué',         'Arabic': 'ماذا'},
    'How':      {'English': 'How',        'French': 'Comment',     'Spanish': 'Cómo',        'Arabic': 'كيف'},
    'You':      {'English': 'You',        'French': 'Toi',         'Spanish': 'Tú',          'Arabic': 'أنت'},
    'Please':   {'English': 'Please',     'French': "S'il te plaît",'Spanish': 'Por favor',   'Arabic': 'من فضلك'},
    'Sorry':    {'English': 'Sorry',      'French': 'Désolé',      'Spanish': 'Lo siento',   'Arabic': 'آسف'},
    'Help':     {'English': 'Help',       'French': 'Aide',        'Spanish': 'Ayuda',       'Arabic': 'مساعدة'},
    'Yes':      {'English': 'Yes',        'French': 'Oui',         'Spanish': 'Sí',          'Arabic': 'نعم'},
    'No':       {'English': 'No',         'French': 'Non',         'Spanish': 'No',          'Arabic': 'لا'},
    'Good':     {'English': 'Good',       'French': 'Bien',        'Spanish': 'Bien',        'Arabic': 'جيد'},
    'Bad':      {'English': 'Bad',        'French': 'Mal',         'Spanish': 'Mal',         'Arabic': 'سيء'},
}

# LSTM thresholds per word
LSTM_THRESHOLDS = {w.capitalize(): 0.70 for w in actions}
LSTM_THRESHOLDS['Hello'] = 0.55
LSTM_THRESHOLDS['Iloveyou'] = 0.60

# ============================================================
# 3. AUDIO ENGINE (Thread-Safe gTTS + Pygame)
# ============================================================

pygame.mixer.init()
tts_queue = queue.Queue()

def tts_worker():
    while True:
        job = tts_queue.get()
        if job is None:
            break
        text, lang_code = job
        try:
            tts = gTTS(text=text, lang=lang_code)
            temp_filename = f"temp_tts_{time.time()}.mp3"
            tts.save(temp_filename)
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            try:
                os.remove(temp_filename)
            except:
                pass
        except Exception as e:
            print(f"TTS Error: {e}")
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak_text(text, lang_name="English"):
    lang_code = LANG_CODES.get(lang_name, 'en')
    tts_queue.put((text, lang_code))

def translate_text(text, target_lang_name):
    if target_lang_name == 'English' or not text.strip():
        return text
    target_code = LANG_CODES.get(target_lang_name, 'en')
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

# ============================================================
# 4. GLOBAL APPLICATION STATE
# ============================================================

practice_state = {
    'alphabet': 'N/A',
    'word': '',
    'sentence': '',
    'language': 'English',
    'is_paused': False
}

learn_state = {
    'target_id': None,
    'target_label': None,
    'detected': False,
    'confidence': 0.0
}

# Progress tracking
progress = {}  # { lesson_id: True/False }
for lesson in ALL_LESSONS:
    progress[lesson['id']] = False

# Internal buffers for practice mode
practice_sequence = []
practice_stab_buffer = []
practice_word_buffer = ""
practice_last_time = time.time()
REGISTRATION_DELAY = 1.5
practice_no_hand_counter = 0

# Internal buffers for learn mode
learn_sequence = []
learn_stab_buffer = []
learn_hold_count = 0

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_rf_prediction(hand_landmarks):
    """Get Random Forest prediction from hand landmarks."""
    data_aux, x_, y_ = [], [], []
    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))
    if len(data_aux) < EXPECTED_FEATURES:
        data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))
    elif len(data_aux) > EXPECTED_FEATURES:
        data_aux = data_aux[:EXPECTED_FEATURES]
    prediction = rf_model.predict([np.asarray(data_aux)])
    try:
        return LABELS_DICT[int(prediction[0])]
    except KeyError:
        return None

# ============================================================
# 6. VIDEO GENERATORS
# ============================================================

def generate_practice_frames():
    """Main practice mode video stream with full Dual-Brain logic."""
    global practice_sequence, practice_stab_buffer, practice_word_buffer
    global practice_last_time, practice_no_hand_counter

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            if practice_state['is_paused']:
                ret, buf = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            predicted_character = None
            phrase_triggered = False
            has_hands = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

            if has_hands:
                practice_no_hand_counter = 0
            else:
                practice_no_hand_counter += 1

            if practice_no_hand_counter > 10:
                practice_sequence.clear()

            # --- Brain 2: LSTM for phrases ---
            if has_hands:
                keypoints = extract_keypoints(results)
                practice_sequence.append(keypoints)
                if len(practice_sequence) > 30:
                    practice_sequence = practice_sequence[-30:]

                current_time = time.time()

                # Initialize cooldown tracker
                if not hasattr(generate_practice_frames, '_lstm_cooldown'):
                    generate_practice_frames._lstm_cooldown = 0

                if len(practice_sequence) == 30 and current_time > generate_practice_frames._lstm_cooldown:
                    input_seq = np.array(practice_sequence)
                    if norm_mean is not None:
                        input_seq = (input_seq - norm_mean) / norm_std
                    res = lstm_model.predict(np.expand_dims(input_seq, axis=0), verbose=0)[0]

                    # ONLY accept the words we need — ignore all others
                    ALLOWED_WORDS = {'hello', 'my', 'name', 'is'}
                    best_word = None
                    best_conf = 0.0
                    for i, w in enumerate(actions):
                        if w in ALLOWED_WORDS and res[i] > best_conf:
                            best_word = w
                            best_conf = res[i]

                    threshold = LSTM_THRESHOLDS.get(best_word.capitalize(), 0.70) if best_word else 0.70

                    if best_word and best_conf > threshold:
                        # Stabilization: must predict same word multiple times
                        if not hasattr(generate_practice_frames, '_last_word'):
                            generate_practice_frames._last_word = None
                            generate_practice_frames._word_count = 0

                        if best_word == generate_practice_frames._last_word:
                            generate_practice_frames._word_count += 1
                        else:
                            generate_practice_frames._last_word = best_word
                            generate_practice_frames._word_count = 1

                        # Accept after 8 consecutive same predictions
                        if generate_practice_frames._word_count >= 8:
                            predicted_character = best_word.capitalize()
                            phrase_triggered = True
                            practice_sequence.clear()
                            generate_practice_frames._last_word = None
                            generate_practice_frames._word_count = 0
                            generate_practice_frames._lstm_cooldown = current_time + 1.5
                    else:
                        # Reset stabilization on low confidence
                        if hasattr(generate_practice_frames, '_last_word'):
                            generate_practice_frames._last_word = None
                            generate_practice_frames._word_count = 0

            # --- Brain 1: Random Forest for letters ---
            if not phrase_triggered and has_hands:
                hands_list = []
                if results.left_hand_landmarks:
                    hands_list.append(results.left_hand_landmarks)
                if results.right_hand_landmarks:
                    hands_list.append(results.right_hand_landmarks)
                for hl in hands_list:
                    pred = get_rf_prediction(hl)
                    if pred is not None:
                        predicted_character = pred
                        break

            # --- Output Logic ---
            if predicted_character:
                current_time = time.time()
                if current_time - practice_last_time < REGISTRATION_DELAY:
                    practice_stab_buffer.clear()
                    phrase_triggered = False

                if phrase_triggered:
                    if current_time - practice_last_time > REGISTRATION_DELAY:
                        target_lang = practice_state['language']
                        translated = GESTURE_TRANSLATIONS.get(predicted_character, {}).get(target_lang, predicted_character)
                        practice_last_time = current_time
                        practice_state['alphabet'] = translated

                        if practice_word_buffer.strip():
                            w = practice_word_buffer.strip()
                            def _speak_phrase(word, lang, gesture):
                                tw = translate_text(word, lang)
                                speak_text(tw, lang)
                                practice_state['sentence'] += tw + " "
                                speak_text(gesture, lang)
                                practice_state['sentence'] += gesture + " "
                            threading.Thread(target=_speak_phrase, args=(w, target_lang, translated), daemon=True).start()
                        else:
                            speak_text(translated, target_lang)
                            practice_state['sentence'] += translated + " "

                        practice_word_buffer = ""
                        practice_state['word'] = ""
                        practice_sequence.clear()
                else:
                    practice_stab_buffer.append(predicted_character)
                    if len(practice_stab_buffer) > 30:
                        practice_stab_buffer.pop(0)

                    if practice_stab_buffer.count(predicted_character) > 20:
                        current_time = time.time()
                        if current_time - practice_last_time > REGISTRATION_DELAY:
                            practice_last_time = current_time
                            practice_state['alphabet'] = predicted_character

                            if predicted_character == ' ':
                                if practice_word_buffer.strip():
                                    target_lang = practice_state['language']
                                    w = practice_word_buffer.strip()
                                    def _speak_space(word, lang):
                                        tw = translate_text(word, lang)
                                        speak_text(tw, lang)
                                        practice_state['sentence'] += tw + " "
                                    threading.Thread(target=_speak_space, args=(w, target_lang), daemon=True).start()
                                practice_word_buffer = ""
                                practice_state['word'] = ""
                            elif predicted_character == '.':
                                if practice_word_buffer.strip():
                                    target_lang = practice_state['language']
                                    w = practice_word_buffer.strip()
                                    def _speak_dot(word, lang):
                                        tw = translate_text(word, lang)
                                        speak_text(tw, lang)
                                        practice_state['sentence'] += tw + ". "
                                    threading.Thread(target=_speak_dot, args=(w, target_lang), daemon=True).start()
                                practice_word_buffer = ""
                                practice_state['word'] = ""
                            else:
                                practice_word_buffer += predicted_character
                                practice_state['word'] = practice_word_buffer

            # Draw landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


def generate_learn_frames():
    """Learning mode video stream — validates a single specific gesture."""
    global learn_sequence, learn_stab_buffer, learn_hold_count

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            target_id = learn_state['target_id']
            if target_id is None:
                ret, buf = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            has_hands = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

            detected = False

            # Check if this is a gesture (LSTM) lesson
            gesture_ids = [w for w in actions if w != 'neutral']
            if target_id in gesture_ids:
                if has_hands:
                    keypoints = extract_keypoints(results)
                    learn_sequence.append(keypoints)
                    if len(learn_sequence) > 30:
                        learn_sequence = learn_sequence[-30:]
                    if len(learn_sequence) == 30:
                        input_seq = np.array(learn_sequence)
                        if norm_mean is not None:
                            input_seq = (input_seq - norm_mean) / norm_std
                        res = lstm_model.predict(np.expand_dims(input_seq, axis=0), verbose=0)[0]
                        max_idx = np.argmax(res)
                        action_name = actions[max_idx].capitalize()
                        threshold = LSTM_THRESHOLDS.get(action_name, 0.70)
                        if action_name.lower() == target_id and res[max_idx] > threshold:
                            detected = True
                            learn_state['confidence'] = float(res[max_idx])
                else:
                    learn_sequence.clear()
            else:
                # Static letter/number — use Random Forest
                if has_hands:
                    hands_list = []
                    if results.left_hand_landmarks:
                        hands_list.append(results.left_hand_landmarks)
                    if results.right_hand_landmarks:
                        hands_list.append(results.right_hand_landmarks)
                    for hl in hands_list:
                        pred = get_rf_prediction(hl)
                        if pred is not None:
                            target_label = LABELS_DICT.get(int(target_id), None)
                            if pred == target_label:
                                learn_stab_buffer.append(pred)
                                if len(learn_stab_buffer) > 15:
                                    learn_stab_buffer.pop(0)
                                if learn_stab_buffer.count(pred) >= 10:
                                    detected = True
                                    learn_state['confidence'] = 1.0
                            else:
                                learn_stab_buffer.clear()
                        break

            if detected and not learn_state['detected']:
                learn_state['detected'] = True
                progress[target_id] = True

            # Draw status overlay
            if learn_state['detected']:
                cv2.putText(frame, "CORRECT!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Show the sign...", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


# ============================================================
# 7. FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    completed = sum(1 for v in progress.values() if v)
    total = len(progress)
    return render_template('index.html', completed=completed, total=total)

@app.route('/learn')
def learn():
    completed = sum(1 for v in progress.values() if v)
    total = len(progress)
    return render_template('learn.html', lessons=ALL_LESSONS, progress=progress,
                         completed=completed, total=total)

@app.route('/learn/<lesson_id>')
def lesson(lesson_id):
    global learn_sequence, learn_stab_buffer
    # Find the lesson
    lesson_data = None
    for l in ALL_LESSONS:
        if l['id'] == lesson_id:
            lesson_data = l
            break
    if not lesson_data:
        return "Lesson not found", 404

    # Reset learn state
    learn_state['target_id'] = lesson_id
    learn_state['target_label'] = lesson_data['label']
    learn_state['detected'] = False
    learn_state['confidence'] = 0.0
    learn_sequence.clear()
    learn_stab_buffer.clear()

    # Find prev/next lessons
    ids = [l['id'] for l in ALL_LESSONS]
    idx = ids.index(lesson_id)
    prev_id = ids[idx - 1] if idx > 0 else None
    next_id = ids[idx + 1] if idx < len(ids) - 1 else None

    return render_template('lesson.html', lesson=lesson_data, prev_id=prev_id, next_id=next_id)

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/phrase')
def phrase_demo():
    return render_template('phrase.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_practice_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_learn')
def video_feed_learn():
    return Response(generate_learn_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify(practice_state)

@app.route('/learn_state')
def get_learn_state():
    return jsonify(learn_state)

@app.route('/progress')
def get_progress():
    completed = sum(1 for v in progress.values() if v)
    total = len(progress)
    return jsonify({'progress': progress, 'completed': completed, 'total': total})

@app.route('/reference/<lesson_id>')
def reference_image(lesson_id):
    """Serve a reference image from the training data."""
    img_path = os.path.join('data', str(lesson_id), '0.jpg')
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    # For gesture lessons, return a placeholder
    return "", 404

@app.route('/command', methods=['POST'])
def handle_command():
    global practice_word_buffer, practice_stab_buffer, practice_sequence
    data = request.json
    action = data.get('action')

    if action == 'reset':
        practice_state['alphabet'] = 'N/A'
        practice_state['word'] = ''
        practice_state['sentence'] = ''
        practice_word_buffer = ""
        practice_stab_buffer = []
        practice_sequence = []
    elif action == 'pause':
        practice_state['is_paused'] = not practice_state['is_paused']
    elif action == 'language':
        practice_state['language'] = data.get('language', 'English')
    elif action == 'speak':
        if practice_state['sentence'].strip():
            speak_text(practice_state['sentence'].strip(), practice_state['language'])
    elif action == 'delete':
        if practice_word_buffer:
            practice_word_buffer = practice_word_buffer[:-1]
            practice_state['word'] = practice_word_buffer
        elif practice_state['sentence']:
            practice_state['sentence'] = practice_state['sentence'].rstrip()
            if practice_state['sentence']:
                parts = practice_state['sentence'].rsplit(' ', 1)
                if len(parts) > 1:
                    practice_state['sentence'] = parts[0] + ' '
                    practice_word_buffer = parts[1]
                    practice_state['word'] = practice_word_buffer
                else:
                    practice_word_buffer = parts[0]
                    practice_state['word'] = practice_word_buffer
                    practice_state['sentence'] = ''
    elif action == 'reset_progress':
        for key in progress:
            progress[key] = False

    return jsonify({"status": "ok", "state": practice_state})

# ============================================================
# PHRASE DEMO — State Machine + Video Feed (100% Random Forest)
# ============================================================

PHRASE_TG1 = ['hello','my','name','is']
PHRASE_DG1 = ['Hello','my','name','is']
PHRASE_TL  = list('Omar')
PHRASE_TG2 = ['how','you']
PHRASE_DG2 = ['How are','you']
PHRASE_GESTURE_STAB = 3
PHRASE_LETTER_STAB = 3

def _phrase_build(parts):
    d='';s=''
    for p in parts:
        if len(p)==1: s+=p
        else:
            if s: d+=' '+s; s=''
            d+=(' ' if d else '')+p
    if s: d+=' '+s
    return d.strip()

phrase_state_data = {
    'step': 0, 'parts': [], 'phase': 'G1', 'complete': False,
    'buf': [], 'lg': None, 'gc': 0, 'cd': 0,
    'll': None, 'lc': 0,
    'live': '', 'conf': 0.0,
    'language': 'English'
}

def _phrase_reset():
    phrase_state_data.update({
        'step': 0, 'parts': [], 'phase': 'G1', 'complete': False,
        'buf': [], 'lg': None, 'gc': 0, 'cd': 0,
        'll': None, 'lc': 0, 'live': '', 'conf': 0.0
    })

def _seq_to_features(seq):
    arr = np.array(seq)
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    mx = np.max(arr, axis=0)
    mn = np.min(arr, axis=0)
    d = np.diff(arr, axis=0)
    vm = np.mean(np.abs(d), axis=0)
    vs = np.std(d, axis=0)
    return np.concatenate([m, s, mx, mn, vm, vs])

def _extract_hands_126(results):
    lh = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

def generate_phrase_frames():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    sd = phrase_state_data

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            has_hands = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # --- GESTURE PHASES ---
            if sd['phase'] in ('G1','G2') and not sd['complete'] and gesture_rf_model:
                tg = PHRASE_TG1 if sd['phase']=='G1' else PHRASE_TG2
                dg = PHRASE_DG1 if sd['phase']=='G1' else PHRASE_DG2

                hands = _extract_hands_126(results)
                sd['buf'].append(hands)
                if len(sd['buf'])>30: sd['buf']=sd['buf'][-30:]

                if len(sd['buf'])==30 and has_hands and now>sd['cd']:
                    feat = _seq_to_features(sd['buf']).reshape(1,-1)
                    pred = gesture_rf_model.predict(feat)[0]
                    proba = gesture_rf_model.predict_proba(feat)[0]
                    conf = float(np.max(proba))
                    sd['live'] = pred; sd['conf'] = conf

                    target = tg[sd['step']]
                    if pred==target and conf>0.50:
                        if pred==sd['lg']: sd['gc']+=1
                        else: sd['lg']=pred; sd['gc']=1
                        if sd['gc']>=PHRASE_GESTURE_STAB:
                            sd['parts'].append(dg[sd['step']])
                            speak_text(dg[sd['step']], sd['language'])
                            sd['step']+=1; sd['buf']=[]; sd['lg']=None; sd['gc']=0
                            sd['cd']=now+1.5
                            if sd['step']>=len(tg):
                                sd['step']=0
                                if sd['phase']=='G1':
                                    sd['phase']='LETTER'
                                else:
                                    sd['complete']=True
                                    speak_text(_phrase_build(sd['parts']), sd['language'])
                    elif pred=='neutral':
                        pass
                    else:
                        sd['lg']=None; sd['gc']=0
                elif not has_hands:
                    sd['live']=''; sd['conf']=0.0

            # --- LETTER PHASE ---
            elif sd['phase']=='LETTER' and not sd['complete']:
                if has_hands and now>sd['cd']:
                    hl = results.right_hand_landmarks or results.left_hand_landmarks
                    if hl:
                        letter = get_rf_prediction(hl)
                        if letter:
                            sd['live']=letter; sd['conf']=1.0
                            exp = PHRASE_TL[sd['step']]
                            if letter.upper()==exp.upper():
                                if letter==sd['ll']: sd['lc']+=1
                                else: sd['ll']=letter; sd['lc']=1
                                if sd['lc']>=PHRASE_LETTER_STAB:
                                    sd['parts'].append(exp)
                                    sd['step']+=1; sd['ll']=None; sd['lc']=0; sd['cd']=now+0.8
                                    if sd['step']>=len(PHRASE_TL):
                                        sd['phase']='G2'; sd['step']=0
                            else:
                                sd['ll']=None; sd['lc']=0
                elif not has_hands:
                    sd['live']=''; sd['ll']=None; sd['lc']=0

            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed_phrase')
def video_feed_phrase():
    return Response(generate_phrase_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/phrase_state')
def get_phrase_state():
    sd = phrase_state_data
    phase = sd['phase']
    mode = 'GESTURE' if phase in ('G1','G2') else 'LETTER'

    if sd['complete']:
        current = 'Complete!'
    elif phase=='G1':
        current = f'Sign: "{PHRASE_TG1[sd["step"]]}"'
    elif phase=='LETTER':
        current = f'Letter: "{PHRASE_TL[sd["step"]]}"'
    elif phase=='G2':
        current = f'Sign: "{PHRASE_TG2[sd["step"]]}"'
    else:
        current = ''

    phrase = _phrase_build(sd['parts'])
    lang = sd['language']
    translation = translate_text(phrase, lang) if phrase and lang != 'English' else phrase

    total = len(PHRASE_TG1)+len(PHRASE_TL)+len(PHRASE_TG2)
    done = len(sd['parts'])

    return jsonify({
        'mode': mode,
        'current_step': current,
        'detection': str(sd['live']) if sd['live'] else '',
        'phrase': phrase,
        'translation': translation or '',
        'complete': sd['complete'],
        'total': total,
        'done': done
    })

@app.route('/phrase_command', methods=['POST'])
def phrase_command():
    data = request.json
    action = data.get('action')
    if action == 'reset':
        _phrase_reset()
    elif action == 'delete':
        sd = phrase_state_data
        if sd['parts']:
            removed = sd['parts'].pop()
            sd['complete'] = False
            # Figure out which phase/step to go back to
            done = len(sd['parts'])
            g1_len = len(PHRASE_TG1)
            letter_len = len(PHRASE_TL)
            if done < g1_len:
                sd['phase'] = 'G1'; sd['step'] = done
            elif done < g1_len + letter_len:
                sd['phase'] = 'LETTER'; sd['step'] = done - g1_len
            else:
                sd['phase'] = 'G2'; sd['step'] = done - g1_len - letter_len
            sd['buf'] = []; sd['lg'] = None; sd['gc'] = 0
            sd['ll'] = None; sd['lc'] = 0; sd['cd'] = 0
    elif action == 'language':
        phrase_state_data['language'] = data.get('language', 'English')
    elif action == 'speak':
        phrase = _phrase_build(phrase_state_data['parts'])
        if phrase:
            lang = phrase_state_data['language']
            translated = translate_text(phrase, lang) if lang != 'English' else phrase
            speak_text(translated, lang)
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  ASL Learning Platform is running!")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
