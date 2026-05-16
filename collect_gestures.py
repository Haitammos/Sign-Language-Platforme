"""
  COLLECT: how, are, you (30 sequences each, 30 frames)
  Saves to MP_Data_Phrase/how/, MP_Data_Phrase/are/, MP_Data_Phrase/you/
  
  Q = record | S = skip | ESC = quit
"""
import os, cv2, numpy as np, mediapipe as mp

DATA_PATH = 'MP_Data_Phrase'
WORDS = ['how', 'are', 'you']
NUM_SEQUENCES = 30
SEQUENCE_LENGTH = 30

INSTRUCTIONS = {
    'how': 'Both hands fists, knuckles together, roll outward',
    'are': 'R-finger from lips forward',
    'you': 'Point index finger forward at camera',
}

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[r.x,r.y,r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

for word in WORDS:
    for seq in range(NUM_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, word, str(seq)), exist_ok=True)

print(f"\n{'='*55}")
print(f"  COLLECT: how, are, you")
print(f"  3 words x 30 sequences")
print(f"  Q = record | S = skip | ESC = quit")
print(f"{'='*55}\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0) as holistic:
    for wi, word in enumerate(WORDS):
        print(f"\n  [{wi+1}/{len(WORDS)}] {word.upper()}: {INSTRUCTIONS[word]}")
        for seq in range(NUM_SEQUENCES):
            check = os.path.join(DATA_PATH, word, str(seq), f'{SEQUENCE_LENGTH-1}.npy')
            if os.path.exists(check): continue

            skip_word = False
            while True:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                if res.left_hand_landmarks: mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if res.right_hand_landmarks: mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                h, w = int(frame.shape[0]), int(frame.shape[1])
                cv2.rectangle(frame, (0,0), (w,80), (0,0,0), -1)
                cv2.putText(frame, f'{word.upper()}: {INSTRUCTIONS[word]}', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,255), 2)
                cv2.putText(frame, f'Sample {seq+1}/{NUM_SEQUENCES}', (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
                cv2.putText(frame, 'Q=record | S=skip | ESC=quit', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
                cv2.imshow('Collect', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'): break
                elif key == ord('s'): skip_word = True; break
                elif key == 27: cap.release(); cv2.destroyAllWindows(); exit(0)
            if skip_word: break

            for fn in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = holistic.process(rgb)
                h, w = int(frame.shape[0]), int(frame.shape[1])
                cv2.rectangle(frame, (0,0), (w,30), (0,0,180), -1)
                cv2.putText(frame, f'REC {word.upper()} [{seq+1}] {fn+1}/{SEQUENCE_LENGTH}', (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
                if res.right_hand_landmarks: mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                cv2.imshow('Collect', frame)
                cv2.waitKey(30)
                np.save(os.path.join(DATA_PATH, word, str(seq), str(fn)), extract_keypoints(res))
            print(f'    {word} [{seq+1}/{NUM_SEQUENCES}] saved')

cap.release()
cv2.destroyAllWindows()
print(f'\n  Done! Run: python train_gesture_rf.py')
