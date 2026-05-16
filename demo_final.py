"""
=============================================================================
  DEMO FINAL — "Hello my name is Omar How are you"
  
  100% Random Forest. Zero LSTM. Dynamic gestures.
  
  Gesture RF: temporal stats (mean/std/velocity) of 126 hand features
  Letter RF: static hand landmarks (model.p)
  
  R = restart | ESC = quit
=============================================================================
"""
import os, sys, cv2, time, pickle, threading
import numpy as np
import mediapipe as mp

# =============================================================================
# 1. LOAD
# =============================================================================
print("[*] Loading...")

class _Up(pickle.Unpickler):
    def find_class(self, m, n):
        if m.startswith('numpy._core'): m = m.replace('numpy._core', 'numpy.core')
        return super().find_class(m, n)

try:
    with open('gesture_rf.p', 'rb') as f:
        gd = pickle.load(f)
    gesture_rf = gd['model']
    WORDS = gd['words']
    print(f"[OK] Gesture RF: {WORDS}")
except:
    print("[!] gesture_rf.p not found! Run train_gesture_rf.py"); sys.exit(1)

try:
    with open('model.p', 'rb') as f:
        rf_model = _Up(f).load()['model']
    print("[OK] Letter RF")
except Exception as e:
    print(f"[!] model.p: {e}"); sys.exit(1)

LABELS = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',14:'O',15:'P',16:'Q',17:'R',18:'S',
    19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
    36:' ',37:'.'
}
print("[OK] Ready!\n")

# =============================================================================
# 2. HELPERS
# =============================================================================
def extract_hands_126(results):
    """Extract left hand (63) + right hand (63) = 126 from Holistic."""
    lh = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

def seq_to_features(seq):
    arr = np.array(seq)
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    mx = np.max(arr, axis=0)
    mn = np.min(arr, axis=0)
    d = np.diff(arr, axis=0)
    vm = np.mean(np.abs(d), axis=0)
    vs = np.std(d, axis=0)
    return np.concatenate([m, s, mx, mn, vm, vs])

def get_letter(hand_lm):
    da, x_, y_ = [], [], []
    for p in hand_lm.landmark: x_.append(p.x); y_.append(p.y)
    for p in hand_lm.landmark: da.append(p.x - min(x_)); da.append(p.y - min(y_))
    if len(da) < 42: da.extend([0]*(42-len(da)))
    elif len(da) > 42: da = da[:42]
    pred = rf_model.predict([np.asarray(da)])[0]
    try: return LABELS[int(pred)]
    except: return None

def build_phrase(parts):
    d='';s=''
    for p in parts:
        if len(p)==1: s+=p
        else:
            if s: d+=' '+s; s=''
            d+=(' ' if d else '')+p
    if s: d+=' '+s
    return d.strip()

def speak_async(text):
    def _s():
        try:
            from gtts import gTTS; import pygame
            pygame.mixer.init()
            fn=f'_tts_{int(time.time())}.mp3'
            gTTS(text=text,lang='en').save(fn)
            pygame.mixer.music.load(fn); pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
            time.sleep(0.3)
            try: pygame.mixer.music.unload()
            except: pass
            try: os.remove(fn)
            except: pass
        except: pass
    threading.Thread(target=_s, daemon=True).start()

# =============================================================================
# 3. STATE
# =============================================================================
TG1 = ['hello','my','name','is']     # Phase 1: gestures
DG1 = ['Hello','my','name','is']
TL  = list('Omar')                  # Phase 2: letters
TG2 = ['how','you']                   # Phase 3: gestures
DG2 = ['How are','you']               # 'how' gesture → "How are"
GESTURE_STAB = 3
LETTER_STAB = 3

class S:
    def __init__(self):
        self.step=0; self.parts=[]; self.phase='G1'; self.complete=False
        self.buf=[]; self.lg=None; self.gc=0; self.cd=0
        self.ll=None; self.lc=0
        self.live=''; self.conf=0.0; self.flash=''; self.ft=0
    def reset(self): self.__init__()
    @property
    def mode(self):
        return 'GESTURE' if self.phase in ('G1','G2') else 'LETTER'

st = S()

# =============================================================================
# 4. MAIN
# =============================================================================
print("="*55)
print("  DEMO: 'Hello my name is Omar How are you'")
print("  100% Random Forest")
print("  R = restart | ESC = quit")
print("="*55+"\n")

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)
if not cap.isOpened(): print("[!] No camera"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fc=0; ft=time.time(); fps=0

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue
        fc+=1; now=time.time()
        fh,fw = int(frame.shape[0]), int(frame.shape[1])
        if now-ft>=1.0: fps=fc; fc=0; ft=now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)

        has_hands = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)

        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # ─── GESTURE PHASES (G1 and G2) ───
        if st.phase in ('G1','G2') and not st.complete:
            tg = TG1 if st.phase=='G1' else TG2
            dg = DG1 if st.phase=='G1' else DG2

            hands = extract_hands_126(results)
            st.buf.append(hands)
            if len(st.buf)>30: st.buf=st.buf[-30:]

            if len(st.buf)==30 and has_hands and now>st.cd:
                feat = seq_to_features(st.buf).reshape(1,-1)
                pred = gesture_rf.predict(feat)[0]
                proba = gesture_rf.predict_proba(feat)[0]
                conf = float(np.max(proba))
                st.live=pred; st.conf=conf

                target = tg[st.step]
                if pred==target and conf>0.50:
                    if pred==st.lg: st.gc+=1
                    else: st.lg=pred; st.gc=1
                    if st.gc>=GESTURE_STAB:
                        st.parts.append(dg[st.step])
                        st.flash=dg[st.step]; st.ft=now
                        print(f"  >>> {dg[st.step]}")
                        speak_async(dg[st.step])
                        st.step+=1; st.buf.clear(); st.lg=None; st.gc=0
                        st.cd=now+1.5
                        if st.step>=len(tg):
                            st.step=0
                            if st.phase=='G1':
                                st.phase='LETTER'
                                print("  >>> LETTER MODE")
                            else:
                                st.complete=True
                                full=build_phrase(st.parts)
                                print(f"\n{'='*55}\n  COMPLETE: {full}\n{'='*55}\n")
                                speak_async(full)
                elif pred=='neutral':
                    pass
                else:
                    st.lg=None; st.gc=0
            elif not has_hands:
                st.live=''; st.conf=0.0

        # ─── LETTER PHASE ───
        elif st.phase=='LETTER' and not st.complete:
            if has_hands and now>st.cd:
                hl = results.right_hand_landmarks or results.left_hand_landmarks
                if hl:
                    letter = get_letter(hl)
                    if letter:
                        st.live=letter; st.conf=1.0
                        exp = TL[st.step]
                        if letter.upper()==exp.upper():
                            if letter==st.ll: st.lc+=1
                            else: st.ll=letter; st.lc=1
                            if st.lc>=LETTER_STAB:
                                st.parts.append(exp)
                                st.flash=exp; st.ft=now
                                print(f"  >>> {exp}")
                                st.step+=1; st.ll=None; st.lc=0; st.cd=now+0.8
                                if st.step>=len(TL):
                                    st.phase='G2'; st.step=0
                                    print("  >>> GESTURE MODE 2")
                        else:
                            st.ll=None; st.lc=0
            elif not has_hands:
                st.live=''; st.ll=None; st.lc=0

        # ─── UI ───
        cv2.rectangle(frame,(0,0),(fw,85),(12,12,12),-1)
        mc=(255,200,50) if st.mode=='GESTURE' else (50,200,255)
        cv2.putText(frame,f"{st.mode}  FPS:{fps}",(15,18),cv2.FONT_HERSHEY_SIMPLEX,0.4,mc,1)
        if not st.complete:
            if st.phase=='G1':
                txt=f'Sign: "{TG1[st.step]}"'
            elif st.phase=='LETTER':
                txt=f'Letter: "{TL[st.step]}"'
            elif st.phase=='G2':
                txt=f'Sign: "{TG2[st.step]}"'
            else:
                txt=''
            cv2.putText(frame,txt,(15,48),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,230,255),2)
        cv2.putText(frame,"R=restart  ESC=quit",(15,78),cv2.FONT_HERSHEY_SIMPLEX,0.33,(60,60,60),1)
        cv2.circle(frame,(fw-20,15),8,(0,255,0) if has_hands else (0,0,200),-1)

        # Detection
        if st.live and not st.complete:
            bx=fw-200
            cv2.rectangle(frame,(bx,90),(fw-8,170),(20,20,20),-1)
            cv2.rectangle(frame,(bx,90),(fw-8,170),(50,50,50),1)
            cv2.putText(frame,"RF:",(bx+8,108),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,100,100),1)
            dc=(0,255,0) if st.conf>0.70 else (0,180,255) if st.conf>0.50 else (100,100,255)
            cv2.putText(frame,str(st.live),(bx+8,138),cv2.FONT_HERSHEY_SIMPLEX,0.85,dc,2)
            cv2.putText(frame,f"{st.conf:.0%}",(bx+130,108),cv2.FONT_HERSHEY_SIMPLEX,0.35,dc,1)
            sc=st.gc if st.mode=='GESTURE' else st.lc
            sm=GESTURE_STAB if st.mode=='GESTURE' else LETTER_STAB
            sp=min(sc/sm,1.0)
            cv2.rectangle(frame,(bx+8,150),(bx+182,160),(40,40,40),-1)
            cv2.rectangle(frame,(bx+8,150),(bx+8+int(174*sp),160),(0,255,0) if sp>=1 else (0,150,200),-1)

        # Flash
        if st.flash and now-st.ft<0.8:
            cv2.putText(frame,f"OK: {st.flash}",(fw//2-80,fh//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        # Bottom
        py=fh-70
        cv2.rectangle(frame,(0,py),(fw,fh),(12,12,12),-1)
        cv2.line(frame,(0,py),(fw,py),(40,40,40),1)
        phrase=build_phrase(st.parts)
        if phrase:
            cv2.putText(frame,phrase,(15,py+28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            total=len(TG1)+len(TL)+len(TG2)
            prog=len(st.parts)/total
            bw=int((fw-30)*prog)
            cv2.rectangle(frame,(15,py+42),(fw-15,py+52),(35,35,35),-1)
            cv2.rectangle(frame,(15,py+42),(15+bw,py+52),(0,200,100),-1)
        else:
            cv2.putText(frame,"Sign your first gesture...",(15,py+28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(70,70,70),1)

        if st.complete:
            ov=frame.copy()
            cv2.rectangle(ov,(0,0),(fw,fh),(0,50,0),-1)
            frame=cv2.addWeighted(ov,0.25,frame,0.75,0)
            cv2.rectangle(frame,(0,0),(fw,85),(0,70,0),-1)
            cv2.putText(frame,"PHRASE COMPLETE!",(fw//2-170,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,100),3)
            cv2.putText(frame,build_phrase(st.parts),(fw//2-170,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        cv2.imshow('ASL Demo - Hello my name is Omar How are you', frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('r'): st.reset(); print("\n  >>> RESTART\n")
        elif key==27: break

cap.release()
cv2.destroyAllWindows()
