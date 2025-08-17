# cricket_drive.py

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# --------- Paths ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- Helper Functions ---------
def angle_between(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def compute_metrics(landmarks, img_w, img_h):
    lm = lambda idx: (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
    left_shoulder = lm(11)
    left_elbow = lm(13)
    left_wrist = lm(15)
    left_hip = lm(23)
    left_knee = lm(25)
    nose = lm(0)
    
    front_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
    spine_angle = angle_between(left_hip, left_shoulder, (left_shoulder[0], left_shoulder[1]-100))
    head_over_knee = abs(nose[0] - left_knee[0])
    
    return {
        "front_elbow_angle": front_elbow_angle,
        "spine_angle": spine_angle,
        "head_over_knee_dist": head_over_knee
    }

def overlay_metrics(frame, metrics):
    cv2.putText(frame, f"Elbow: {int(metrics['front_elbow_angle'])}°", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Spine: {int(metrics['spine_angle'])}°", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Head-Knee Dist: {int(metrics['head_over_knee_dist'])}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if metrics['front_elbow_angle'] > 110:
        cv2.putText(frame, "Good elbow elevation ✅", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Elbow low ❌", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if metrics['head_over_knee_dist'] < 30:
        cv2.putText(frame, "Head over knee ✅", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Head too forward ❌", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

# --------- Video Processing Function ---------
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, 'annotated_video.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_metrics = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            metrics = compute_metrics(results.pose_landmarks.landmark, width, height)
            all_metrics.append(metrics)
            frame = overlay_metrics(frame, metrics)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_elbow = np.mean([m['front_elbow_angle'] for m in all_metrics])
    avg_spine = np.mean([m['spine_angle'] for m in all_metrics])
    avg_head_knee = np.mean([m['head_over_knee_dist'] for m in all_metrics])

    evaluation = {
        "Footwork": round(8 if avg_head_knee < 30 else 5, 1),
        "Head Position": round(8 if avg_head_knee < 30 else 5, 1),
        "Swing Control": round(8 if avg_elbow > 110 else 5, 1),
        "Balance": round(8 if 60 < avg_spine < 120 else 5, 1),
        "Follow-through": round(7, 1),
        "Feedback": {
            "Footwork": "Good alignment" if avg_head_knee < 30 else "Needs improvement",
            "Head Position": "Head over front knee ✅" if avg_head_knee < 30 else "Head too forward ❌",
            "Swing Control": "Elbow angle sufficient" if avg_elbow > 110 else "Elbow too low",
            "Balance": "Spine lean ok" if 60 < avg_spine < 120 else "Adjust posture",
            "Follow-through": "Follow-through reasonable"
        }
    }

    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=4)

    return evaluation

# --------- Flask Routes ---------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    video_path = os.path.join(OUTPUT_DIR, video_file.filename)
    video_file.save(video_path)

    evaluation = analyze_video(video_path)
    return jsonify(evaluation)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# --------- Run Flask App ---------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
