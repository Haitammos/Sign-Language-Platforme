import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame, OptionMenu
from PIL import Image, ImageTk
import threading
import time
import warnings
import os
import queue
from gtts import gTTS
import pygame
from deep_translator import GoogleTranslator

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# --- 1. Load LSTM Action Recognition Model (18 Words + neutral) ---
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

# Load normalization params
try:
    norm_mean = np.load('norm_mean.npy')
    norm_std = np.load('norm_std.npy')
    norm_std[norm_std == 0] = 1
except:
    norm_mean = None
    norm_std = None

sequence = []
lstm_thresholds = {w.capitalize(): 0.70 for w in actions}
lstm_thresholds['Hello'] = 0.55
lstm_thresholds['Iloveyou'] = 0.60


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

# --- 2. Load RF Alphabet Model (Letters) ---
with open('./model.p', 'rb') as f:
    model_dict = RenameUnpickler(f).load()
model = model_dict['model']

# Mediapipe setup (Upgraded to Holistic to capture body/face for LSTM actions)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1)

# Initialize Pygame Mixer for Audio (Replaces pyttsx3)
pygame.mixer.init()

# ---------------- MULTI-LANGUAGE TRANSLATION CONFIG ----------------
# Hardcoded instantaneous translation for Gestures
GESTURE_TRANSLATIONS = {
    'Hello': {'English': 'Hello', 'French': 'Bonjour', 'Spanish': 'Hola', 'Arabic': 'مرحبا'},
    'Thanks': {'English': 'Thanks', 'French': 'Merci', 'Spanish': 'Gracias', 'Arabic': 'شكرا'},
    'Iloveyou': {'English': 'I love you', 'French': "Je t'aime", 'Spanish': 'Te quiero', 'Arabic': 'أنا أحبك'}
}

# Language codes for gTTS and deep-translator
LANG_CODES = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'Arabic': 'ar'
}

# UI Translation Map
UI_TEXT = {
    'English': {'gesture': 'Current Gesture:', 'word': 'Current Word:', 'sentence': 'Current Sentence:', 'reset': 'Reset Sentence', 'pause': 'Pause', 'speak': 'Speak Sentence', 'play': 'Play', 'title': 'Sign Language Dual-Brain AI'},
    'French': {'gesture': 'Geste Courant:', 'word': 'Mot Courant:', 'sentence': 'Phrase Courante:', 'reset': 'Réinitialiser', 'pause': 'Pause', 'speak': 'Énoncer la Phrase', 'play': 'Jouer', 'title': 'IA Bilingue de Langue des Signes'},
    'Spanish': {'gesture': 'Gesto Actual:', 'word': 'Palabra Actual:', 'sentence': 'Oración Actual:', 'reset': 'Reiniciar Oración', 'pause': 'Pausa', 'speak': 'Hablar Oración', 'play': 'Reproducir', 'title': 'IA Bilingüe de Lenguaje de Señas'},
    'Arabic': {'gesture': 'الإيماءة الحالية:', 'word': 'الكلمة الحالية:', 'sentence': 'الجملة الحالية:', 'reset': 'إعادة تعيين 		 الجملة', 'pause': 'إيقاف مؤقت', 'speak': 'نطق الجملة', 'play': 'تشغيل', 'title': 'الذكاء الاصطناعي المزدوج للغة الإشارة'}
}
# -------------------------------------------------------------------

# label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ',
    37: '.'
}
expected_features = 42

# Helper function to extract 1662 dimensions for LSTM
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Initialize buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
no_hand_counter = 0

# Create a thread-safe queue for TTS messages
tts_queue = queue.Queue()

# Single dedicated worker thread for TTS with gTTS
def tts_worker():
    while True:
        job = tts_queue.get()
        if job is None: break
        text, lang_code = job
        try:
            # Generate mp3 with Google TTS and play via pygame
            tts = gTTS(text=text, lang=lang_code)
            temp_filename = f"temp_tts_{time.time()}.mp3"
            tts.save(temp_filename)
            
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
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

# Speak text function queues a tuple (text, language)
def speak_text(text, lang_name="English"):
    lang_code = LANG_CODES.get(lang_name, 'en')
    tts_queue.put((text, lang_code))

def translate_text(text, target_lang_name):
    if target_lang_name == 'English' or not text.strip():
        return text
    target_code = LANG_CODES.get(target_lang_name, 'en')
    try:
        translator = GoogleTranslator(source='auto', target=target_code)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

# GUI Setup
root = tk.Tk()
root.title("Sign Language to Speech Conversion")
root.geometry("1300x650")
root.configure(bg="#2c2f33")
root.resizable(False, False)

# Variables for GUI
current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
is_paused = StringVar(value="False")

# Title
title_label = Label(root, text="Sign Language Dual-Brain AI", font=("Arial", 28, "bold"), fg="#ffffff", bg="#2c2f33")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Layout Frames
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=500, height=400)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)

content_frame = Frame(root, bg="#2c2f33")
content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))

button_frame = Frame(root, bg="#2c2f33")
button_frame.grid(row=3, column=1, pady=(10, 20),padx=(10, 20), sticky="n")

# Video feed
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

# Labels
lbl_gesture = Label(content_frame, text="Current Gesture/Alphabet:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33")
lbl_gesture.pack(anchor="w", pady=(0, 10))
Label(content_frame, textvariable=current_alphabet, font=("Arial", 24, "bold"), fg="#1abc9c", bg="#2c2f33").pack(anchor="center")

lbl_word = Label(content_frame, text="Current Word:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33")
lbl_word.pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_word, font=("Arial", 20), fg="#f39c12", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

lbl_sentence = Label(content_frame, text="Current Sentence:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33")
lbl_sentence.pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_sentence, font=("Arial", 20), fg="#9b59b6", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

def reset_sentence():
    global word_buffer, sentence, stabilization_buffer, sequence
    word_buffer = ""
    sentence = ""
    stabilization_buffer = []
    sequence = []
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

def toggle_pause():
    lang = selected_language.get()
    texts = UI_TEXT.get(lang, UI_TEXT['English'])
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text=texts['play'])
    else:
        is_paused.set("False")
        pause_button.config(text=texts['pause'])

# Buttons
reset_button = Button(button_frame, text="Reset Sentence", font=("Arial", 16), command=reset_sentence, bg="#e74c3c", fg="#ffffff", relief="flat", height=2, width=14)
reset_button.grid(row=0, column=0, padx=10)

pause_button = Button(button_frame, text="Pause", font=("Arial", 16), command=toggle_pause, bg="#3498db", fg="#ffffff", relief="flat", height=2, width=12)
pause_button.grid(row=0, column=1, padx=10)

speak_button = Button(button_frame, text="Speak Sentence", font=("Arial", 16), command=lambda: speak_text(current_sentence.get(), selected_language.get()), bg="#27ae60", fg="#ffffff", relief="flat", height=2, width=14)
speak_button.grid(row=0, column=2, padx=10)

# Language Selector
selected_language = StringVar(value="English")

def update_ui_language(*args):
    lang = selected_language.get()
    texts = UI_TEXT.get(lang, UI_TEXT['English'])
    title_label.config(text=texts['title'])
    lbl_gesture.config(text=texts['gesture'])
    lbl_word.config(text=texts['word'])
    lbl_sentence.config(text=texts['sentence'])
    reset_button.config(text=texts['reset'])
    pause_button.config(text=texts['play'] if is_paused.get() == "True" else texts['pause'])
    speak_button.config(text=texts['speak'])

selected_language.trace("w", update_ui_language)

lang_menu = OptionMenu(button_frame, selected_language, "English", "French", "Spanish", "Arabic")
lang_menu.config(font=("Arial", 14), bg="#e67e22", fg="#ffffff", relief="flat", height=2, width=10)
lang_menu.grid(row=0, column=3, padx=10)

# Video Capture
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

last_registered_time = time.time()
registration_delay = 1.5

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, sequence, no_hand_counter

    ret, frame = cap.read()
    if not ret:
        return

    if is_paused.get() == "True":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process Holistics (Face, Body, and Both Hands)
    results = holistic.process(frame_rgb)
    
    predicted_character = None
    
    # 1. Ask Brain 2 (LSTM) for phrases first by feeding it the sequence
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    
    if len(sequence) > 30:
        sequence.pop(0)

    # Note: Using TensorFlow prediction on every frame can be computationally expensive
    # and causes hallucinations if no hands are in the frame!
    phrase_triggered = False
    has_hands = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    
    if has_hands:
        no_hand_counter = 0
    else:
        no_hand_counter += 1

    # Only clear the sequence if hands have been missing for more than 10 consecutive frames
    # This prevents the action from being broken by a single dropped tracking frame!
    if no_hand_counter > 10:
        sequence.clear()

    if len(sequence) == 30 and has_hands:
        # LSTM Prediction
        res = lstm_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        max_idx = np.argmax(res)
        action_name = actions[max_idx].capitalize()

        # Use specific threshold based on the action
        current_threshold = lstm_thresholds.get(action_name, 0.85)

        if res[max_idx] > current_threshold:
            predicted_character = action_name
            phrase_triggered = True

    # 2. Ask Brain 1 (Random Forest) if Brain 2 didn't confidently detect a phrase
    if not phrase_triggered:
        multi_hand_landmarks = []
        # Fallback to the hands data from holistic to mimic the old mp_hands structure
        if results.left_hand_landmarks:
            multi_hand_landmarks.append(results.left_hand_landmarks)
        if results.right_hand_landmarks:
            multi_hand_landmarks.append(results.right_hand_landmarks)
            
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                if len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))
                elif len(data_aux) > expected_features:
                    data_aux = data_aux[:expected_features]

                # Random Forest Prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                break # Just read one hand directly if both are present in static mode

    # 3. Output logic (Split between Dynamic Phrases and Static Letters)
    if predicted_character:
        current_time = time.time()
        
        # --- GLOBAL COOLDOWN ---
        # Prevent any new letters or phrases from building up for 1.5 seconds after a successful output.
        # This completely stops "c" from leaking after "hello", or "0" from leaking during "i love you".
        if current_time - last_registered_time < registration_delay:
            stabilization_buffer.clear()
            phrase_triggered = False

        if phrase_triggered:
            # Phrases bypass the 15-frame hold requirement because dynamic gestures only peak briefly!
            if current_time - last_registered_time > registration_delay:
                # INSTANT GESTURE TRANSLATION
                target_lang = selected_language.get()
                translated_gesture = GESTURE_TRANSLATIONS.get(predicted_character, {}).get(target_lang, predicted_character)
                
                stable_char = translated_gesture
                last_registered_time = current_time
                current_alphabet.set(stable_char)

                # Word building logic for phrases
                if word_buffer.strip():
                    target_lang = selected_language.get()
                    word_to_trans = word_buffer.strip()

                    def bg_translate_phrase(w, lang, gesture):
                        global sentence
                        tw = translate_text(w, lang)
                        speak_text(tw, lang)
                        sentence += tw + " "
                        speak_text(gesture, lang)
                        sentence += gesture + " "
                        root.after(0, lambda: current_sentence.set(sentence.strip()))
                    
                    threading.Thread(target=bg_translate_phrase, args=(word_to_trans, target_lang, stable_char), daemon=True).start()

                else:
                    target_lang = selected_language.get()
                    speak_text(stable_char, target_lang)
                    sentence += stable_char + " "
                    current_sentence.set(sentence.strip())
                word_buffer = ""
                current_word.set("N/A")
                sequence.clear() # Clear the sequence buffer so it doesn't double-trigger
                
        else:
            # Letters require the user to hold the pose stably
            stabilization_buffer.append(predicted_character)
            if len(stabilization_buffer) > 30:
                stabilization_buffer.pop(0)

            # Requires multiple consistent frames to stabilize a letter (Increased to 25 to defeat motion leaks)
            if stabilization_buffer.count(predicted_character) > 25:
                if current_time - last_registered_time > registration_delay:
                    stable_char = predicted_character
                    last_registered_time = current_time
                    current_alphabet.set(stable_char)

                    # Word building logic for Letters
                    if stable_char == ' ':
                        if word_buffer.strip():
                            target_lang = selected_language.get()
                            word_to_trans = word_buffer.strip()

                            def bg_translate_space(w, lang):
                                global sentence
                                tw = translate_text(w, lang)
                                speak_text(tw, lang)
                                sentence += tw + " "
                                root.after(0, lambda: current_sentence.set(sentence.strip()))
                            
                            threading.Thread(target=bg_translate_space, args=(word_to_trans, target_lang), daemon=True).start()
                        word_buffer = ""
                        current_word.set("N/A")
                    elif stable_char == '.':
                        if word_buffer.strip():
                            target_lang = selected_language.get()
                            word_to_trans = word_buffer.strip()
                            
                            def bg_translate_dot(w, lang):
                                global sentence
                                tw = translate_text(w, lang)
                                speak_text(tw, lang)
                                sentence += tw + "."
                                root.after(0, lambda: current_sentence.set(sentence.strip()))

                            threading.Thread(target=bg_translate_dot, args=(word_to_trans, target_lang), daemon=True).start()
                        word_buffer = ""
                        current_word.set("N/A")
                    else:
                        word_buffer += stable_char
                        current_word.set(word_buffer)

    # Draw Landmarks for Feedback
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # (Optional: Draw Pose landmarks if needed, but omitted for visual clarity)

    cv2.putText(frame, f"Output: {current_alphabet.get()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Update GUI Video Label
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

# Start processing frames
process_frame()
root.mainloop()
