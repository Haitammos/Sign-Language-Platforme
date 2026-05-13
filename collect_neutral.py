"""
=====================================================
  COLLECT NEUTRAL CLASS DATA (2 minutes)
  Just sit normally - don't sign anything!
=====================================================
"""
import os, cv2, numpy as np, mediapipe as mp

DATA_PATH = 'MP_Data'
NUM_SEQUENCES = 30
SEQUENCE_LENGTH = 30
CAMERA_INDEX = 1

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[r.x,r.y,r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

# Create folders
for seq in range(NUM_SEQUENCES):
    os.makedirs(os.path.join(DATA_PATH, 'neutral', str(seq)), exist_ok=True)

print(f"\n{'='*50}")
print(f"  NEUTRAL CLASS DATA COLLECTION")
print(f"  Just sit normally. Move your hands a little")
print(f"  but DON'T do any sign language gesture!")
print(f"  Press Q to start each recording.")
print(f"{'='*50}\n")

cap = cv2.VideoCapture(CAMERA_INDEX)

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1) as holistic:
    for sequence in range(NUM_SEQUENCES):
        # Check if already exists
        check = os.path.join(DATA_PATH, 'neutral', str(sequence), f'{SEQUENCE_LENGTH-1}.npy')
        if os.path.exists(check):
            print(f'  [SKIP] Sequence {sequence+1} already exists')
            continue

        # Wait screen
        while True:
            ret, frame = cap.read()
            if not ret: continue
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            cv2.putText(frame, 'NEUTRAL - Just sit normally', (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.putText(frame, f'Sample {sequence+1}/{NUM_SEQUENCES} | Press Q to record',
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow('Neutral Collection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Record
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret: continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, f'RECORDING neutral [{sequence+1}/{NUM_SEQUENCES}] Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            bar_w = int(frame.shape[1] * (frame_num+1) / SEQUENCE_LENGTH)
            cv2.rectangle(frame, (0, 35), (bar_w, 40), (0, 255, 0), -1)
            cv2.imshow('Neutral Collection', frame)
            cv2.waitKey(30)

            keypoints = extract_keypoints(results)
            np.save(os.path.join(DATA_PATH, 'neutral', str(sequence), str(frame_num)), keypoints)

        print(f'  [neutral] Sequence {sequence+1}/{NUM_SEQUENCES} saved.')

cap.release()
cv2.destroyAllWindows()
print(f'\n  Done! Now run: python train_lstm.py')
