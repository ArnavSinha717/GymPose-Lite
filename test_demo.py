"""Test the full pipeline on a static image and save the multi-panel output."""

import yaml
import numpy as np
import cv2
import torch

from models.gympose_lite import GymPoseLite
from feedback.angle_utils import get_joint_angles
from feedback.form_checker import check_form
from viz.feature_maps import extract_feature_maps, render_feature_grid
from viz.heatmap_overlay import render_heatmap_grid
from viz.pipeline_display import draw_skeleton, build_display


def preprocess_frame(frame, input_size=(256, 192)):
    h, w = frame.shape[:2]
    target_aspect = input_size[1] / input_size[0]
    frame_aspect = w / h

    if frame_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x1 = (w - new_w) // 2
        crop = frame[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_aspect)
        y1 = (h - new_h) // 2
        crop = frame[y1:y1 + new_h]

    resized = cv2.resize(crop, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    display_crop = resized.copy()

    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, display_crop


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GymPoseLite(num_keypoints=17, use_gnn=True, gnn_hidden=64, pretrained=False).to(device)
    ckpt = torch.load("checkpoints/gym_best.pth", map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Try to load a gym dataset image as test
    import glob
    test_images = glob.glob("datasets/gym/JIM_DATA29/**/images/train/*.jpg", recursive=True)
    if not test_images:
        test_images = glob.glob("datasets/coco/val2017/*.jpg")

    if not test_images:
        print("No test images found!")
        return

    img_path = test_images[0]
    print(f"Testing with: {img_path}")

    frame_rgb = cv2.imread(img_path)
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    exercise = "squat"

    # Step 1: Preprocess
    input_tensor, preprocessed = preprocess_frame(frame_rgb)
    input_tensor = input_tensor.to(device)

    # Step 2: Feature maps
    early_maps, deep_maps = extract_feature_maps(model, input_tensor)
    early_grid = render_feature_grid(early_maps, n=4, panel_size=(100, 100))
    deep_grid = render_feature_grid(deep_maps, n=4, panel_size=(100, 100))
    feature_grid = np.vstack([early_grid, deep_grid])

    # Step 3: Full model
    with torch.no_grad():
        heatmaps, coords = model(input_tensor, exercise=exercise)

    heatmaps_np = heatmaps[0].cpu().numpy()
    coords_np = coords[0].cpu().numpy()

    # Step 4: Heatmap viz
    heatmap_grid = render_heatmap_grid(heatmaps_np)

    # Step 5: Skeleton + feedback
    kp_pixels = coords_np.copy()
    kp_pixels[:, 0] *= preprocessed.shape[1]
    kp_pixels[:, 1] *= preprocessed.shape[0]

    angles = get_joint_angles(kp_pixels)
    feedback = check_form(kp_pixels, exercise)

    severities = [s for _, s in feedback]
    if "bad" in severities:
        overall = "bad"
    elif "warning" in severities:
        overall = "warning"
    else:
        overall = "good"

    skeleton_frame = draw_skeleton(preprocessed, coords_np, severity=overall)

    # Step 6: Multi-panel display
    display = build_display(
        raw_frame=frame_rgb,
        preprocessed=preprocessed,
        feature_grid=feature_grid,
        heatmap_grid=heatmap_grid,
        skeleton_frame=skeleton_frame,
        feedback_list=feedback,
        exercise=exercise,
        angles=angles,
    )

    # Save output
    display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    out_path = "demo_output.png"
    cv2.imwrite(out_path, display_bgr)
    print(f"Saved demo output to: {out_path}")
    print(f"Image size: {display.shape[1]}x{display.shape[0]}")
    print(f"\nFeedback ({exercise}):")
    for text, severity in feedback:
        print(f"  [{severity}] {text}")
    print(f"\nAngles:")
    for name, val in angles.items():
        print(f"  {name}: {val:.1f} deg")


if __name__ == "__main__":
    main()
