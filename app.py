"""
Intelligent Accident Detection System - Flask Web App
Real-time detection with live browser dashboard
"""

import cv2
import numpy as np
import time
import threading
import json
import os
import base64
import random
from datetime import datetime
from collections import deque
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EVIDENCE_FOLDER'] = 'evidence'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
state = {
    'running': False,
    'paused': False,
    'source': None,           # 'webcam' | 'video'
    'show_heatmap': True,
    'show_boxes': True,
    'frame_count': 0,
    'accidents_detected': 0,
    'alerts_sent': 0,
    'fps': 0,
    'last_detection': None,
    'severity': 'NONE',
    'confidence': 0,
    'vehicle_count': 0,
    'accident_detected': False,
    'accident_log': [],
    'response_times': [],
    'last_alert_time': 0,
    'alert_cooldown': 5,
}

frame_buffer = {'frame': None, 'lock': threading.Lock()}
frame_history = deque(maxlen=10)

# ─────────────────────────────────────────────
# DETECTION ENGINE
# ─────────────────────────────────────────────

def detect_vehicles_simulated(frame):
    """Simulate vehicle detection (replace with YOLO if available)"""
    h, w = frame.shape[:2]
    vehicles = []
    
    # Try YOLO first
    try:
        from ultralytics import YOLO
        if not hasattr(detect_vehicles_simulated, '_yolo'):
            detect_vehicles_simulated._yolo = YOLO('yolov8n.pt')
        results = detect_vehicles_simulated._yolo(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        vehicles.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'center': ((x1+x2)//2, (y1+y2)//2),
                            'area': (x2-x1)*(y2-y1),
                            'type': ['car','motorcycle','bus','truck'][cls % 4]
                        })
        return vehicles
    except:
        pass

    # Fallback: simulation
    n = random.randint(1, 5)
    for i in range(n):
        x = random.randint(50, w - 200)
        y = random.randint(50, h - 200)
        bw = random.randint(80, 180)
        bh = random.randint(60, 140)
        vehicles.append({
            'bbox': (x, y, x+bw, y+bh),
            'confidence': random.uniform(0.72, 0.97),
            'center': (x + bw//2, y + bh//2),
            'area': bw * bh,
            'type': random.choice(['car', 'truck', 'motorcycle'])
        })
    return vehicles


def calculate_overlap(b1, b2):
    x_left = max(b1[0], b2[0])
    y_top  = max(b1[1], b2[1])
    x_right  = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    if x_right > x_left and y_bottom > y_top:
        return (x_right - x_left) * (y_bottom - y_top)
    return 0


def calculate_motion(frame):
    frame_history.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if len(frame_history) < 2:
        return 0.0
    try:
        flow = cv2.calcOpticalFlowFarneback(
            frame_history[-2], frame_history[-1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))
    except:
        diff = cv2.absdiff(frame_history[-2], frame_history[-1])
        return float(np.mean(diff)) / 10.0


def classify_severity(vehicles, motion, overlap):
    score = 0
    score += min(len(vehicles) * 10, 30)
    score += min(overlap / 100, 30)
    score += min(motion * 2, 20)
    score = min(score, 100)

    if score >= 75:
        level = 'CRITICAL'
        color = (0, 0, 220)
    elif score >= 45:
        level = 'MAJOR'
        color = (0, 140, 255)
    else:
        level = 'MINOR'
        color = (0, 220, 220)
    return level, score, color


def calculate_confidence(det_conf, sev_score, motion, vehicle_count):
    f1 = det_conf * 40
    f2 = (sev_score / 100) * 30
    f3 = min(motion / 5, 1.0) * 20
    f4 = min(vehicle_count / 5, 1.0) * 10
    total = min(f1 + f2 + f3 + f4, 99)
    if total >= 85:
        level = 'HIGH'
    elif total >= 65:
        level = 'MEDIUM'
    else:
        level = 'LOW'
    return total, level


def draw_heatmap(frame, vehicles, severity_score):
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    intensity = severity_score / 100.0

    for v in vehicles:
        x1, y1, x2, y2 = v['bbox']
        cx, cy = v['center']
        radius = max(min(abs(x2-x1), abs(y2-y1)) // 2, 15)
        y_g, x_g = np.ogrid[:h, :w]
        dist = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
        sigma = max(radius / 2.5, 5)
        spot = np.exp(-(dist**2) / (2 * sigma**2))
        spot[dist > radius * 2] = 0
        heatmap += spot * v['confidence'] * intensity

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    hm8 = (heatmap * 255).astype(np.uint8)
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[:, :, 2] = hm8
    colored[:, :, 1] = np.clip(255 - np.abs(hm8.astype(int) - 128) * 2, 0, 255).astype(np.uint8)
    colored[:, :, 0] = np.clip(255 - hm8 * 2, 0, 255).astype(np.uint8)

    mask = heatmap > 0.1
    frame[mask] = cv2.addWeighted(frame, 0.4, colored, 0.6, 0)[mask]
    return frame


def draw_vehicle_boxes(frame, vehicles, accident_detected, sev_color):
    for i, v in enumerate(vehicles):
        x1, y1, x2, y2 = v['bbox']
        color = sev_color if accident_detected else (0, 200, 80)
        thickness = 3 if accident_detected else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{v['type'].upper()} {v['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw collision lines between overlapping pairs
        if accident_detected:
            cx, cy = v['center']
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

    if accident_detected and len(vehicles) >= 2:
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                overlap = calculate_overlap(vehicles[i]['bbox'], vehicles[j]['bbox'])
                if overlap > 100:
                    cv2.arrowedLine(frame,
                        vehicles[i]['center'], vehicles[j]['center'],
                        (0, 0, 255), 2, tipLength=0.2)
    return frame


def draw_overlay(frame, vehicles, accident_detected, severity, sev_score, confidence, conf_level, motion, fps):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 44), (15, 15, 25), -1)
    cv2.putText(frame, "ACCIDENT DETECTION SYSTEM", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)
    ts = datetime.now().strftime("%H:%M:%S  %d/%m/%Y")
    cv2.putText(frame, ts, (w - 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    # Status badge
    if accident_detected:
        sev_colors = {'MINOR': (0, 200, 200), 'MAJOR': (0, 140, 255), 'CRITICAL': (0, 0, 220)}
        sc = sev_colors.get(severity, (255, 255, 255))
        cv2.rectangle(frame, (w//2 - 120, 50), (w//2 + 120, 90), sc, -1)
        cv2.putText(frame, f"⚠  ACCIDENT — {severity}", (w//2 - 110, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
    else:
        cv2.rectangle(frame, (w//2 - 80, 50), (w//2 + 80, 86), (0, 150, 60), -1)
        cv2.putText(frame, "MONITORING", (w//2 - 65, 76),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    # Bottom info bar
    cv2.rectangle(frame, (0, h - 50), (w, h), (15, 15, 25), -1)

    info_items = [
        (f"FPS: {fps:.1f}", (80, 200, 80)),
        (f"Vehicles: {len(vehicles)}", (100, 180, 255)),
        (f"Confidence: {confidence:.1f}% [{conf_level}]", (255, 220, 80)),
        (f"Severity Score: {sev_score:.1f}", (255, 140, 60)),
        (f"Motion: {motion:.2f}", (180, 180, 255)),
    ]

    x_pos = 12
    for text, color in info_items:
        cv2.putText(frame, text, (x_pos, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        x_pos += len(text) * 9 + 20

    return frame


def save_evidence(frame, severity, alert_id):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(app.config['EVIDENCE_FOLDER'], f"accident_{alert_id}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


# ─────────────────────────────────────────────
# VIDEO PROCESSING THREAD
# ─────────────────────────────────────────────

cap_holder = {'cap': None}

def processing_thread():
    global state
    cap = cap_holder['cap']
    fps_timer = time.time()
    fps_count = 0

    while state['running']:
        if state['paused'] or cap is None:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            if state['source'] == 'video':
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (1280, 720))
        t0 = time.time()

        # Detect
        vehicles = detect_vehicles_simulated(frame)
        motion = calculate_motion(frame)

        # Collision check
        accident_detected = False
        max_overlap = 0
        avg_conf = 0

        if len(vehicles) >= 2:
            overlaps = []
            for i in range(len(vehicles)):
                for j in range(i+1, len(vehicles)):
                    ov = calculate_overlap(vehicles[i]['bbox'], vehicles[j]['bbox'])
                    if ov > 0:
                        overlaps.append(ov)
            if overlaps:
                max_overlap = max(overlaps)
                if max_overlap > 500 or motion > 8:
                    accident_detected = True

        avg_conf = np.mean([v['confidence'] for v in vehicles]) if vehicles else 0.0

        severity, sev_score, sev_color = classify_severity(vehicles, motion, max_overlap)
        if not accident_detected:
            severity = 'NONE'
            sev_score = 0

        confidence, conf_level = calculate_confidence(avg_conf, sev_score, motion, len(vehicles))

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            state['fps'] = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        # Draw heatmap
        display = frame.copy()
        if state['show_heatmap'] and accident_detected:
            display = draw_heatmap(display, vehicles, sev_score)

        # Draw boxes
        if state['show_boxes']:
            display = draw_vehicle_boxes(display, vehicles, accident_detected, sev_color)

        # Draw overlay
        display = draw_overlay(display, vehicles, accident_detected, severity,
                               sev_score, confidence, conf_level, motion, state['fps'])

        # Alert
        now = time.time()
        if accident_detected and severity in ['MAJOR', 'CRITICAL']:
            if now - state['last_alert_time'] > state['alert_cooldown']:
                alert_id = f"ALT-{int(now)}"
                evidence_path = save_evidence(frame, severity, alert_id)
                alert = {
                    'id': alert_id,
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'severity': severity,
                    'confidence': round(confidence, 1),
                    'vehicles': len(vehicles),
                    'evidence': os.path.basename(evidence_path)
                }
                state['accident_log'].insert(0, alert)
                state['accident_log'] = state['accident_log'][:50]
                state['alerts_sent'] += 1
                state['last_alert_time'] = now

        # Update state
        state['frame_count'] += 1
        state['vehicle_count'] = len(vehicles)
        state['accident_detected'] = accident_detected
        state['severity'] = severity
        state['confidence'] = round(confidence, 1)
        state['sev_score'] = round(sev_score, 1)
        state['motion'] = round(motion, 3)
        state['conf_level'] = conf_level
        if accident_detected:
            state['accidents_detected'] += 1

        # Encode frame for streaming
        _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_buffer['lock']:
            frame_buffer['frame'] = jpeg.tobytes()

        process_time = time.time() - t0
        sleep_time = max(0, (1/30) - process_time)
        time.sleep(sleep_time)

    with frame_buffer['lock']:
        frame_buffer['frame'] = None


def generate_frames():
    while True:
        with frame_buffer['lock']:
            frame = frame_buffer['frame']

        if frame is None:
            # Send placeholder
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No video source — select Webcam or Upload Video",
                        (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            frame = jpeg.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1/30)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'running': state['running'],
        'paused': state['paused'],
        'fps': round(state.get('fps', 0), 1),
        'frame_count': state['frame_count'],
        'vehicle_count': state['vehicle_count'],
        'accident_detected': state['accident_detected'],
        'severity': state['severity'],
        'sev_score': state.get('sev_score', 0),
        'confidence': state['confidence'],
        'conf_level': state.get('conf_level', 'LOW'),
        'motion': state.get('motion', 0),
        'accidents_detected': state['accidents_detected'],
        'alerts_sent': state['alerts_sent'],
        'show_heatmap': state['show_heatmap'],
        'show_boxes': state['show_boxes'],
        'source': state['source'],
        'accident_log': state['accident_log'][:10],
    })

@app.route('/api/start_webcam', methods=['POST'])
def start_webcam():
    stop_processing()
    cam_id = int(request.json.get('camera_id', 0))
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return jsonify({'success': False, 'error': 'Cannot open webcam'})
    cap_holder['cap'] = cap
    state['source'] = 'webcam'
    state['running'] = True
    state['paused'] = False
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    return jsonify({'success': True})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file'})
    f = request.files['video']
    if f.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})
    ext = f.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'success': False, 'error': 'Invalid file type'})

    stop_processing()
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return jsonify({'success': False, 'error': 'Cannot open video'})

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = total / fps if fps > 0 else 0

    cap_holder['cap'] = cap
    state['source'] = 'video'
    state['running'] = True
    state['paused'] = False
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()

    return jsonify({'success': True, 'filename': filename,
                    'frames': total, 'fps': round(fps, 1), 'duration': round(dur, 1)})

@app.route('/api/control', methods=['POST'])
def control():
    action = request.json.get('action')
    if action == 'pause':
        state['paused'] = not state['paused']
    elif action == 'stop':
        stop_processing()
    elif action == 'toggle_heatmap':
        state['show_heatmap'] = not state['show_heatmap']
    elif action == 'toggle_boxes':
        state['show_boxes'] = not state['show_boxes']
    elif action == 'demo_minor':
        _force_demo('MINOR')
    elif action == 'demo_major':
        _force_demo('MAJOR')
    elif action == 'demo_critical':
        _force_demo('CRITICAL')
    return jsonify({'success': True, 'state': {
        'paused': state['paused'],
        'show_heatmap': state['show_heatmap'],
        'show_boxes': state['show_boxes'],
    }})

def _force_demo(severity):
    state['accident_detected'] = True
    state['severity'] = severity
    state['sev_score'] = {'MINOR': 35, 'MAJOR': 60, 'CRITICAL': 90}[severity]
    state['confidence'] = random.uniform(72, 96)

def stop_processing():
    state['running'] = False
    time.sleep(0.3)
    if cap_holder['cap']:
        cap_holder['cap'].release()
        cap_holder['cap'] = None
    with frame_buffer['lock']:
        frame_buffer['frame'] = None

@app.route('/evidence/<filename>')
def evidence_file(filename):
    return send_from_directory(app.config['EVIDENCE_FOLDER'], filename)

@app.route('/api/clear_log', methods=['POST'])
def clear_log():
    state['accident_log'] = []
    return jsonify({'success': True})


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('evidence', exist_ok=True)
    print("🚀 Starting Accident Detection System...")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
