"""Process a video file through the GymPose-Lite pipeline.

Runs person detection + pose estimation + form analysis on each frame
and writes an output MP4 with the 8-panel pipeline display.

Usage:
  python process_video.py --input video.mp4 --checkpoint checkpoints/distill_test.pth --use-cv
"""

import argparse
import yaml
import numpy as np
import cv2
import torch
import torchvision
from tqdm import tqdm

from models.gympose_lite import GymPoseLite
from feedback.angle_utils import get_joint_angles
from feedback.form_checker import check_form
from feedback.cv_channels import compute_cv_channels, build_6ch_input, cv_channels_to_display
from feedback.rep_counter import RepCounter
from feedback.rep_scorer import RepScorer
from viz.feature_maps import extract_feature_maps, render_feature_grid
from viz.heatmap_overlay import render_heatmap_grid
from viz.pipeline_display import draw_skeleton, build_display


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def detect_person(frame_rgb, detector, device, threshold=0.5):
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


def main():
    parser = argparse.ArgumentParser(description="Process video through GymPose-Lite")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="", help="Output video path")
    parser.add_argument("--checkpoint", default="checkpoints/distill_test.pth")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exercise", default="squat", choices=["squat", "pushup", "deadlift"])
    parser.add_argument("--detect-every", type=int, default=3, help="Run detector every N frames")
    parser.add_argument("--use-cv", action="store_true", help="Use 6-channel input (RGB + CV)")
    args = parser.parse_args()

    if not args.output:
        base = args.input.rsplit(".", 1)[0]
        args.output = f"{base}_processed.mp4"

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pose model
    in_channels = 6 if args.use_cv else 3
    model = GymPoseLite(
        num_keypoints=cfg["dataset"]["num_keypoints"],
        use_gnn=cfg["model"]["use_gnn"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        pretrained=False,
        in_channels=in_channels,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load person detector
    print("Loading person detector...")
    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    ).to(device)
    detector.eval()

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input: {total_frames} frames, {fps:.1f} FPS")

    # State
    writer = None
    smoothed_coords = None
    last_bbox = None
    ema_alpha = cfg["demo"]["ema_alpha"]
    rep_counter = RepCounter(args.exercise)
    rep_scorer = RepScorer(args.exercise)

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # --- Person detection ---
        run_detect = last_bbox is None or (args.detect_every > 0 and frame_idx % args.detect_every == 0)
        if run_detect:
            bbox = detect_person(frame_rgb, detector, device)
            if bbox is not None:
                last_bbox = bbox

        raw_display = frame_rgb.copy()
        if last_bbox is not None:
            crop, crop_box = crop_person(frame_rgb, last_bbox)
            cv2.rectangle(raw_display,
                          (crop_box[0], crop_box[1]),
                          (crop_box[2], crop_box[3]),
                          (0, 255, 0), 3)
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

        # --- Classical CV channels ---
        edges, skin, grad = compute_cv_channels(preprocessed)
        canny_vis, skin_vis, sobel_vis = cv_channels_to_display(edges, skin, grad)

        # --- Normalize and build model input ---
        img = preprocessed.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        if args.use_cv:
            combined = build_6ch_input(img, edges, skin, grad)
            input_tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        else:
            input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        # --- Feature maps ---
        early_maps, deep_maps = extract_feature_maps(model, input_tensor)
        early_grid = render_feature_grid(early_maps, n=4, panel_size=(100, 100))
        deep_grid = render_feature_grid(deep_maps, n=4, panel_size=(100, 100))
        feature_grid = np.vstack([early_grid, deep_grid])

        # --- Pose inference ---
        with torch.no_grad():
            heatmaps, coords = model(input_tensor, exercise=args.exercise)

        heatmaps_np = heatmaps[0].cpu().numpy()
        coords_np = coords[0].cpu().numpy()

        # --- Temporal smoothing ---
        if smoothed_coords is None:
            smoothed_coords = coords_np.copy()
        else:
            smoothed_coords = ema_alpha * coords_np + (1 - ema_alpha) * smoothed_coords

        # --- Heatmaps ---
        heatmap_grid = render_heatmap_grid(heatmaps_np)

        # --- Angles, feedback, reps ---
        kp_pixels = smoothed_coords.copy()
        kp_pixels[:, 0] *= preprocessed.shape[1]
        kp_pixels[:, 1] *= preprocessed.shape[0]

        angles = get_joint_angles(kp_pixels)
        feedback = check_form(kp_pixels, args.exercise)

        rep_state = rep_counter.update(angles)
        rep_scorer.update(angles, feedback, rep_state)
        score_summary = rep_scorer.get_summary()

        severities = [s for _, s in feedback]
        overall = "bad" if "bad" in severities else ("warning" if "warning" in severities else "good")

        skeleton_frame = draw_skeleton(preprocessed, smoothed_coords, severity=overall)

        # --- 8-panel display ---
        display = build_display(
            raw_frame=raw_display,
            preprocessed=preprocessed,
            feature_grid=feature_grid,
            heatmap_grid=heatmap_grid,
            skeleton_frame=skeleton_frame,
            feedback_list=feedback,
            exercise=args.exercise,
            angles=angles,
            canny_vis=canny_vis,
            skin_vis=skin_vis,
            sobel_vis=sobel_vis,
            rep_info=rep_state,
            score_summary=score_summary,
        )

        # Initialize writer on first frame
        if writer is None:
            out_h, out_w = display.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))
            print(f"Output: {out_w}x{out_h} @ {fps:.1f} FPS -> {args.output}")

        display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        writer.write(display_bgr)

    cap.release()
    if writer:
        writer.release()
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
