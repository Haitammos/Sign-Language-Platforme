"""
=====================================================
  COLLECT WORD DATA FOR LSTM TRAINING
  Records MediaPipe keypoints for each ASL word sign
=====================================================
"""
import os
import cv2
import numpy as np
import mediapipe as mp

# ─── CONFIG ─────────────────────────────────────────
# Words to train (existing 3 + new ones)
WORDS = [
    'hello', 'thanks', 'iloveyou',   # Already trained
    'hi', 'my', 'name', 'is',        # For "Hi my name is..."
    'what', 'how', 'you',            # For questions
    'please', 'sorry', 'help',       # Polite phrases
    'yes', 'no', 'good', 'bad'       # Basic responses
]

DATA_PATH = os.path.join('MP_Data')   # Folder to store keypoint sequences
NUM_SEQUENCES = 30                     # 30 video samples per word
SEQUENCE_LENGTH = 30                   # 30 frames per video sample
CAMERA_INDEX = 1                       # DroidCam = 1 (change to 0 for built-in webcam)

# ─── MEDIAPIPE SETUP ────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract 1662-dim feature vector from MediaPipe Holistic results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# ─── CREATE FOLDERS ─────────────────────────────────
for word in WORDS:
    for seq in range(NUM_SEQUENCES):
        folder = os.path.join(DATA_PATH, word, str(seq))
        os.makedirs(folder, exist_ok=True)

print(f"\n{'='*50}")
print(f"  ASL Word Data Collection")
print(f"  Words: {len(WORDS)} | Sequences: {NUM_SEQUENCES} | Frames: {SEQUENCE_LENGTH}")
print(f"{'='*50}\n")

# ─── COLLECT DATA ───────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1) as holistic:
    
    for word_idx, word in enumerate(WORDS):
        # Check if data already exists for this word
        existing = 0
        for seq in range(NUM_SEQUENCES):
            check_path = os.path.join(DATA_PATH, word, str(seq), f'{SEQUENCE_LENGTH - 1}.npy')
            if os.path.exists(check_path):
                existing += 1
        
        if existing == NUM_SEQUENCES:
            print(f'[SKIP] "{word}" already has {NUM_SEQUENCES} sequences collected.')
            continue
        
        start_seq = existing  # Resume from where we left off
        
        for sequence in range(start_seq, NUM_SEQUENCES):
            # ── Wait screen: show what word to sign ──
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                cv2.putText(frame, f'WORD: "{word.upper()}"', (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(frame, f'Sample {sequence + 1}/{NUM_SEQUENCES}  |  Word {word_idx + 1}/{len(WORDS)}',
                           (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.putText(frame, 'Press Q to start recording', (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
                
                cv2.imshow('ASL Word Collection', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            # ── Record 30 frames of keypoints ──
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                
                # Draw landmarks
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Status bar
                progress = int((frame_num + 1) / SEQUENCE_LENGTH * 100)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(frame, f'RECORDING "{word.upper()}" [{sequence+1}/{NUM_SEQUENCES}] Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Progress bar
                bar_width = int(frame.shape[1] * progress / 100)
                cv2.rectangle(frame, (0, 45), (bar_width, 50), (0, 255, 0), -1)
                
                cv2.imshow('ASL Word Collection', frame)
                cv2.waitKey(30)
                
                # Save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, word, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
            
            print(f'  [{word}] Sequence {sequence + 1}/{NUM_SEQUENCES} saved.')
        
        print(f'[DONE] "{word}" complete!\n')

cap.release()
cv2.destroyAllWindows()
print(f'\n{"="*50}')
print(f'  Data collection complete!')
print(f'  Next: Run train_lstm.py to train the model')
print(f'{"="*50}')
