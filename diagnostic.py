"""
Quick diagnostic: see what both models are detecting in real-time.
Shows LSTM predictions, confidence levels, and RF predictions side by side.
"""
import os, cv2, pickle, numpy as np, mediapipe as mp, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization

# --- Load LSTM ---
actions = np.array(['hello','thanks','iloveyou','hi','my','name','is',
                    'what','how','you','please','sorry','help','yes','no','good','bad'])

model = Sequential()
model.add(Input(shape=(30, 1662)))
model.add(LSTM(64, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))
model.load_weights('action.h5')

# Load normalization
norm_mean = np.load('norm_mean.npy')
norm_std = np.load('norm_std.npy')
norm_std[norm_std == 0] = 1

# --- Load RF ---
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

with open('./model.p', 'rb') as f:
    model_dict = RenameUnpickler(f).load()
rf_model = model_dict['model']

LABELS_DICT = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
    19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
    36:' ',37:'.'
}

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

# --- Run ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1)
sequence = []

print("\n" + "="*60)
print("  DIAGNOSTIC MODE")
print("  Watch the window. Press Q to quit.")
print("="*60 + "\n")

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        has_hands = results.left_hand_landmarks or results.right_hand_landmarks

        # --- LSTM ---
        lstm_word = "---"
        lstm_conf = 0.0
        if has_hands:
            kp = extract_keypoints(results)
            sequence.append(kp)
            if len(sequence) > 30: sequence = sequence[-30:]
            if len(sequence) == 30:
                inp = np.array(sequence)
                inp = (inp - norm_mean) / norm_std
                res = model.predict(np.expand_dims(inp, axis=0), verbose=0)[0]
                idx = np.argmax(res)
                lstm_word = actions[idx].capitalize()
                lstm_conf = res[idx]
        else:
            sequence.clear()

        # --- RF ---
        rf_letter = "---"
        if has_hands:
            hl = results.right_hand_landmarks or results.left_hand_landmarks
            if hl:
                rf_letter = get_rf_pred(hl) or "---"

        # --- Display ---
        h = frame.shape[0]
        # Black panel at bottom
        cv2.rectangle(frame, (0, h-120), (frame.shape[1], h), (0,0,0), -1)
        
        # LSTM info
        color = (0,255,0) if lstm_conf > 0.70 else (0,165,255) if lstm_conf > 0.40 else (0,0,255)
        cv2.putText(frame, f'LSTM: {lstm_word} ({lstm_conf:.0%})', (10, h-85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Confidence bar
        bar_w = int(lstm_conf * 300)
        cv2.rectangle(frame, (10, h-70), (310, h-55), (50,50,50), -1)
        cv2.rectangle(frame, (10, h-70), (10 + bar_w, h-55), color, -1)

        # RF info
        cv2.putText(frame, f'RF Letter: {rf_letter}', (10, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # Hands status
        status = "HANDS DETECTED" if has_hands else "NO HANDS"
        s_color = (0,255,0) if has_hands else (0,0,255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('DIAGNOSTIC - Press Q to quit', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
