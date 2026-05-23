import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import warnings
import queue
import os
import io
import base64

warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("[WARN] deep-translator not installed. Translation disabled.")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("[WARN] gTTS not installed. Non-English TTS disabled.")

try:
    import nltk
    from nltk.corpus import words as nltk_words
    nltk.download('words', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import brown
    from collections import defaultdict, Counter
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARN] nltk not installed. Word suggestions disabled.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sl2ts-secret-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.p')
model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    max_num_hands=1
)

LABELS_DICT = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
    19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
    36:' ',37:'.'
}
EXPECTED_FEATURES = 42

bigrams = defaultdict(Counter)
word_set = set()

if NLTK_AVAILABLE:
    try:
        brown_words = [w.lower() for w in brown.words() if w.isalpha()]
        word_set = set(brown_words)
        for i in range(len(brown_words) - 1):
            bigrams[brown_words[i]][brown_words[i+1]] += 1
        print(f"[INFO] Loaded {len(word_set)} words, {len(bigrams)} bigrams")
    except Exception as e:
        print(f"[WARN] NLTK corpus load failed: {e}")

def get_word_completions(partial: str, n=4):
    """Return top-n words that start with partial."""
    if not partial or not NLTK_AVAILABLE:
        return []
    p = partial.lower()
    matches = [w for w in word_set if w.startswith(p) and len(w) > len(p)]
    matches.sort(key=lambda w: -bigrams.get('the', Counter()).get(w, 0))
    return matches[:n]

def get_next_word_suggestions(last_word: str, n=4):
    """Return top-n likely next words after last_word."""
    if not last_word or not NLTK_AVAILABLE:
        return []
    lw = last_word.lower()
    if lw in bigrams:
        top = bigrams[lw].most_common(n)
        return [w for w, _ in top]
    return []

state_lock = threading.Lock()
state = {
    'stabilization_buffer': [],
    'stable_char': None,
    'word_buffer': '',
    'sentence': '',
    'last_registered_time': time.time(),
    'registration_delay': 1.5,
    'paused': False,
    'language': 'en',          
    'total_frames': 0,
    'detected_frames': 0,
    'char_timestamps': [],     
    'confidence': 0.0,
    'session_chars': 0,
    'session_words': 0,
    'session_sentences': 0,
}

LANG_CODES = {'en': 'en', 'hi': 'hi', 'gu': 'gu'}

def translate_text(text: str, target_lang: str) -> str:
    if not text.strip() or target_lang == 'en' or not TRANSLATION_AVAILABLE:
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        print(f"[WARN] Translation error: {e}")
        return text

def get_tts_audio_b64(text: str, lang: str) -> str:
    """Return base64-encoded mp3 for non-English TTS, or empty string."""
    if not text.strip() or not GTTS_AVAILABLE or lang == 'en':
        return ''
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"[WARN] gTTS error: {e}")
        return ''

def compute_cpm() -> float:
    now = time.time()
    with state_lock:
        state['char_timestamps'] = [t for t in state['char_timestamps'] if now - t <= 60]
        return len(state['char_timestamps'])   

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_lock = threading.Lock()
latest_frame = None
stop_event = threading.Event()

def inference_loop():
    global latest_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        with state_lock:
            state['total_frames'] += 1
            paused = state['paused']

        if paused:
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.03)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            with state_lock:
                state['detected_frames'] += 1

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux, x_, y_ = [], [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) < EXPECTED_FEATURES:
                    data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))
                elif len(data_aux) > EXPECTED_FEATURES:
                    data_aux = data_aux[:EXPECTED_FEATURES]

                arr = np.asarray(data_aux).reshape(1, -1)
                proba = model.predict_proba(arr)[0]
                top_idx = int(np.argmax(proba))
                confidence = float(proba[top_idx])
                pred_label = model.classes_[top_idx]
                predicted_character = LABELS_DICT[int(pred_label)]

                with state_lock:
                    state['confidence'] = round(confidence * 100, 1)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                with state_lock:
                    buf = state['stabilization_buffer']
                    buf.append(predicted_character)
                    if len(buf) > 30:
                        buf.pop(0)

                    if buf.count(predicted_character) > 25:
                        now = time.time()
                        if now - state['last_registered_time'] > state['registration_delay']:
                            state['last_registered_time'] = now
                            state['stable_char'] = predicted_character
                            state['char_timestamps'].append(now)
                            state['session_chars'] += 1

                            if predicted_character == ' ':
                                if state['word_buffer'].strip():
                                    state['sentence'] += state['word_buffer'] + ' '
                                    state['session_words'] += 1
                                state['word_buffer'] = ''
                            elif predicted_character == '.':
                                if state['word_buffer'].strip():
                                    state['sentence'] += state['word_buffer'] + '.'
                                    state['session_words'] += 1
                                state['word_buffer'] = ''
                                state['session_sentences'] += 1
                            else:
                                state['word_buffer'] += predicted_character

                            _emit_state_update()

        with state_lock:
            char_display = state['stable_char'] or '-'
            conf_display = state['confidence']

        cv2.putText(frame, f"Sign: {char_display}  Conf: {conf_display}%",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.01)

def _emit_state_update():
    """Must be called with state_lock held."""
    word_buf = state['word_buffer']
    sentence = state['sentence']
    lang = state['language']

    completions = get_word_completions(word_buf) if word_buf else []
    last_word = sentence.rstrip('. ').split()[-1] if sentence.split() else ''
    next_words = get_next_word_suggestions(last_word) if last_word and not word_buf else []
    suggestions = completions if completions else next_words

    socketio.emit('state_update', {
        'char': state['stable_char'],
        'word': word_buf,
        'sentence': sentence.strip(),
        'confidence': state['confidence'],
        'suggestions': suggestions,
        'language': lang,
    })

inf_thread = threading.Thread(target=inference_loop, daemon=True)
inf_thread.start()

def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    with state_lock:
        total = state['total_frames']
        detected = state['detected_frames']
    cpm = compute_cpm()
    with state_lock:
        return jsonify({
            'cpm': cpm,
            'confidence': state['confidence'],
            'detection_rate': round((detected / total * 100), 1) if total else 0,
            'session_chars': state['session_chars'],
            'session_words': state['session_words'],
            'session_sentences': state['session_sentences'],
        })

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.json
    text = data.get('text', '')
    lang = data.get('lang', 'en')
    translated = translate_text(text, lang)
    audio_b64 = get_tts_audio_b64(translated, lang)
    return jsonify({'translated': translated, 'audio': audio_b64})

@socketio.on('set_language')
def on_set_language(data):
    lang = data.get('lang', 'en')
    with state_lock:
        state['language'] = lang

@socketio.on('toggle_pause')
def on_toggle_pause():
    with state_lock:
        state['paused'] = not state['paused']
        emit('paused', {'paused': state['paused']})

@socketio.on('reset')
def on_reset():
    with state_lock:
        state['word_buffer'] = ''
        state['sentence'] = ''
        state['stable_char'] = None
        state['stabilization_buffer'] = []
        state['session_chars'] = 0
        state['session_words'] = 0
        state['session_sentences'] = 0
        state['char_timestamps'] = []
        state['confidence'] = 0.0
    emit('state_update', {
        'char': None, 'word': '', 'sentence': '',
        'confidence': 0, 'suggestions': [], 'language': state['language']
    }, broadcast=True)

@socketio.on('apply_suggestion')
def on_apply_suggestion(data):
    word = data.get('word', '')
    with state_lock:
        state['word_buffer'] = word
        _emit_state_update()

import atexit
def cleanup():
    stop_event.set()
    cap.release()
atexit.register(cleanup)

if __name__ == '__main__':
    print("\n🤟 Sign2Text Web Server starting...")
    print("   Open http://localhost:5000 in your browser\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
