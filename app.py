"""GymPose-Lite Web Application.

Upload a video, select an exercise, get back:
- Main video: skeleton overlay + rep counter + form feedback
- Optional pipeline videos: Canny edges, skin mask, Sobel gradient, heatmaps, CNN features
"""

import os
import uuid
import subprocess
import threading
import yaml
import numpy as np
import cv2
import torch
import torchvision
from flask import Flask, render_template, request, jsonify, send_from_directory

from models.gympose_lite import GymPoseLite
from feedback.angle_utils import get_joint_angles
from feedback.form_checker import check_form
from feedback.cv_channels import compute_cv_channels, build_6ch_input, cv_channels_to_display
from feedback.rep_counter import RepCounter
from feedback.rep_scorer import RepScorer
from viz.feature_maps import extract_feature_maps, render_feature_grid
from viz.heatmap_overlay import render_heatmap_grid, render_heatmap_overlay
from viz.pipeline_display import draw_skeleton, SEVERITY_COLORS, SCORE_COLORS

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = None
model = None
detector = None
cfg = None
jobs = {}


def load_models():
    global device, model, detector, cfg
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading pose model...")
    model = GymPoseLite(
        num_keypoints=cfg["dataset"]["num_keypoints"],
        use_gnn=cfg["model"]["use_gnn"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        pretrained=False,
        in_channels=6,
    ).to(device)
    ckpt = torch.load("checkpoints/distill_test.pth", map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print("Loading person detector...")
    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    ).to(device)
    detector.eval()
    print("Models loaded.")


def detect_person(frame_rgb, threshold=0.5):
    img_tensor = torchvision.transforms.functional.to_tensor(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        dets = detector(img_tensor)[0]
    mask = (dets["labels"] == 1) & (dets["scores"] > threshold)
    if not mask.any():
        return None
    boxes = dets["boxes"][mask].cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[np.argmax(areas)].astype(int)


def crop_person(frame, bbox, padding=0.2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * padding), int(bh * padding)
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_main_overlay(frame_rgb, keypoints, feedback_list, exercise, angles,
                      rep_info, score_summary, crop_box=None):
    """Draw skeleton + feedback text on the full original frame."""
    display = frame_rgb.copy()
    h, w = display.shape[:2]

    # Draw skeleton on full frame using crop_box to map coordinates
    if crop_box is not None:
        cx1, cy1, cx2, cy2 = crop_box
        cw, ch = cx2 - cx1, cy2 - cy1
    else:
        cx1, cy1 = 0, 0
        cw, ch = w, h

    # Determine severity color
    severities = [s for _, s in feedback_list]
    if "bad" in severities:
        overall = "bad"
    elif "warning" in severities:
        overall = "warning"
    else:
        overall = "good"
    color = SEVERITY_COLORS.get(overall, (0, 255, 0))

    # Draw bones
    bones = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
             (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for i, j in bones:
        x1 = int(cx1 + keypoints[i][0] * cw)
        y1_pt = int(cy1 + keypoints[i][1] * ch)
        x2 = int(cx1 + keypoints[j][0] * cw)
        y2_pt = int(cy1 + keypoints[j][1] * ch)
        cv2.line(display, (x1, y1_pt), (x2, y2_pt), color, 3)

    for k in range(17):
        x = int(cx1 + keypoints[k][0] * cw)
        y = int(cy1 + keypoints[k][1] * ch)
        cv2.circle(display, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(display, (x, y), 5, color, 2)

    # Dark overlay bar at top for text
    bar_h = 90
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

    # Exercise + rep count
    rep_text = f"{exercise.upper()}"
    if rep_info:
        rep_text += f"  |  Reps: {rep_info['rep_count']}  |  {rep_info['state']}"
    cv2.putText(display, rep_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)

    # Feedback lines
    x_offset = 15
    y_offset = 55
    for text, severity in feedback_list[:3]:
        fc = SEVERITY_COLORS.get(severity, (200, 200, 200))
        cv2.putText(display, text, (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc, 1)
        x_offset += len(text) * 10 + 20

    # Rep history dots at top right
    if score_summary and score_summary["rep_history"]:
        dot_x = w - 30
        for rep in reversed(score_summary["rep_history"]):
            sc = SCORE_COLORS.get(rep["score"], (150, 150, 150))
            cv2.circle(display, (dot_x, 75), 8, sc, -1)
            cv2.putText(display, str(rep["rep_num"]), (dot_x - 4, 79),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            dot_x -= 25

    return display


def reencode_h264(input_path):
    """Re-encode video to H.264 for browser playback."""
    h264_path = input_path.replace(".mp4", "_h264.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart", h264_path,
    ], capture_output=True)
    if os.path.exists(h264_path):
        os.replace(h264_path, input_path)


def process_video(input_path, job_id, exercise):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Could not open video"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        # Output writers — main + optional pipeline videos
        prefix = os.path.join(OUTPUT_DIR, job_id)
        writers = {}
        smoothed_coords = None
        last_bbox = None
        ema_alpha = cfg["demo"]["ema_alpha"]
        rep_counter = RepCounter(exercise)
        rep_scorer = RepScorer(exercise)
        detect_every = 3
        out_size = None  # Set on first frame

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            if out_size is None:
                out_size = (w, h)
                small_size = (384, 288)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers["main"] = cv2.VideoWriter(f"{prefix}_main.mp4", fourcc, fps, out_size)
                writers["canny"] = cv2.VideoWriter(f"{prefix}_canny.mp4", fourcc, fps, small_size)
                writers["skin"] = cv2.VideoWriter(f"{prefix}_skin.mp4", fourcc, fps, small_size)
                writers["sobel"] = cv2.VideoWriter(f"{prefix}_sobel.mp4", fourcc, fps, small_size)
                writers["heatmap"] = cv2.VideoWriter(f"{prefix}_heatmap.mp4", fourcc, fps, small_size)
                writers["features"] = cv2.VideoWriter(f"{prefix}_features.mp4", fourcc, fps, small_size)

            # Person detection
            if frame_idx % detect_every == 0 or last_bbox is None:
                bbox = detect_person(frame_rgb)
                if bbox is not None:
                    last_bbox = bbox

            crop_box = None
            if last_bbox is not None:
                crop, crop_box = crop_person(frame_rgb, last_bbox)
            else:
                target_aspect = 192 / 256
                if w / h > target_aspect:
                    new_w = int(h * target_aspect)
                    x1 = (w - new_w) // 2
                    crop = frame_rgb[:, x1:x1 + new_w]
                else:
                    new_h = int(w / target_aspect)
                    y1 = (h - new_h) // 2
                    crop = frame_rgb[y1:y1 + new_h]

            preprocessed = cv2.resize(crop, (192, 256), interpolation=cv2.INTER_LINEAR)

            # CV channels
            edges, skin, grad = compute_cv_channels(preprocessed)
            canny_vis, skin_vis, sobel_vis = cv_channels_to_display(edges, skin, grad)

            # Model input
            img = preprocessed.astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            combined = build_6ch_input(img, edges, skin, grad)
            input_tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            # Feature maps
            early_maps, deep_maps = extract_feature_maps(model, input_tensor)
            early_grid = render_feature_grid(early_maps, n=4, panel_size=(96, 72))
            deep_grid = render_feature_grid(deep_maps, n=4, panel_size=(96, 72))
            feature_grid = np.vstack([early_grid, deep_grid])

            # Inference
            with torch.no_grad():
                heatmaps, coords = model(input_tensor, exercise=exercise)

            heatmaps_np = heatmaps[0].cpu().numpy()
            coords_np = coords[0].cpu().numpy()

            # Smoothing
            if smoothed_coords is None:
                smoothed_coords = coords_np.copy()
            else:
                smoothed_coords = ema_alpha * coords_np + (1 - ema_alpha) * smoothed_coords

            # Angles + feedback + reps
            kp_pixels = smoothed_coords.copy()
            kp_pixels[:, 0] *= preprocessed.shape[1]
            kp_pixels[:, 1] *= preprocessed.shape[0]

            angles = get_joint_angles(kp_pixels)
            feedback = check_form(kp_pixels, exercise)
            rep_state = rep_counter.update(angles)
            rep_scorer.update(angles, feedback, rep_state)
            score_summary = rep_scorer.get_summary()

            # Main video: skeleton on original frame
            main_frame = draw_main_overlay(
                frame_rgb, smoothed_coords, feedback, exercise, angles,
                rep_state, score_summary, crop_box)
            writers["main"].write(cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR))

            # Pipeline videos (small, resized)
            small = lambda img: cv2.resize(img, (384, 288), interpolation=cv2.INTER_LINEAR)
            writers["canny"].write(cv2.cvtColor(small(canny_vis), cv2.COLOR_RGB2BGR))
            writers["skin"].write(cv2.cvtColor(small(skin_vis), cv2.COLOR_RGB2BGR))
            writers["sobel"].write(cv2.cvtColor(small(sobel_vis), cv2.COLOR_RGB2BGR))

            # Heatmap overlay on preprocessed image
            hm_overlay = render_heatmap_overlay(preprocessed, heatmaps_np, alpha=0.5)
            writers["heatmap"].write(cv2.cvtColor(small(hm_overlay), cv2.COLOR_RGB2BGR))

            writers["features"].write(cv2.cvtColor(small(feature_grid), cv2.COLOR_RGB2BGR))

            jobs[job_id]["progress"] = int((frame_idx + 1) / total_frames * 100)

        cap.release()
        for w in writers.values():
            w.release()

        # Re-encode all to H.264
        for suffix in ["main", "canny", "skin", "sobel", "heatmap", "features"]:
            reencode_h264(f"{prefix}_{suffix}.mp4")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["outputs"] = {
            "main": f"{job_id}_main.mp4",
            "canny": f"{job_id}_canny.mp4",
            "skin": f"{job_id}_skin.mp4",
            "sobel": f"{job_id}_sobel.mp4",
            "heatmap": f"{job_id}_heatmap.mp4",
            "features": f"{job_id}_features.mp4",
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        import traceback
        traceback.print_exc()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    video = request.files["video"]
    exercise = request.form.get("exercise", "squat")

    job_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(video.filename)[1] or ".mp4"
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    video.save(input_path)

    jobs[job_id] = {"status": "processing", "progress": 0, "outputs": {}}

    thread = threading.Thread(target=process_video, args=(input_path, job_id, exercise))
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/static/processed/<filename>")
def serve_processed(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
