"""Multi-panel demo layout for showcasing the image processing pipeline.

Arranges 8 panels into a single window (2x4 grid):
  Row 1: Input Frame | Canny Edge Detection | HSV Skin Segmentation | Sobel Gradient
  Row 2: Heatmaps    | Skeleton Overlay      | Form + Rep Count      | CNN Features
"""

import numpy as np
import cv2

from feedback.angle_utils import get_joint_angles
from feedback.form_checker import check_form

# Colors for severity levels
SEVERITY_COLORS = {
    "good": (0, 200, 0),       # Green
    "info": (200, 200, 0),     # Cyan
    "warning": (0, 200, 200),  # Yellow
    "bad": (0, 0, 200),        # Red
}

# Score colors for rep history dots
SCORE_COLORS = {
    "good": (0, 220, 0),
    "ok": (0, 200, 200),
    "bad": (0, 0, 220),
}

# COCO skeleton bones for drawing
SKELETON_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(image, keypoints, confidence=None, threshold=0.3, severity="good"):
    """Draw skeleton overlay on image with color-coded bones."""
    img = image.copy()
    color = SEVERITY_COLORS.get(severity, (0, 255, 0))

    kp = keypoints  # (17, 2) normalized [0, 1]
    h, w = img.shape[:2]

    # Draw bones
    for i, j in SKELETON_BONES:
        if confidence is not None:
            if confidence[i] < threshold or confidence[j] < threshold:
                continue
        x1, y1 = int(kp[i][0] * w), int(kp[i][1] * h)
        x2, y2 = int(kp[j][0] * w), int(kp[j][1] * h)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Draw joints
    for k in range(17):
        if confidence is not None and confidence[k] < threshold:
            continue
        x, y = int(kp[k][0] * w), int(kp[k][1] * h)
        cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 4, color, 1)

    return img


def draw_feedback_panel(size, feedback_list, exercise, angles=None,
                        rep_info=None, score_summary=None):
    """Draw panel with form feedback, rep count, and per-rep scores."""
    panel = np.zeros((size[1], size[0], 3), dtype=np.uint8) + 30  # Dark gray bg

    # Exercise title + rep count
    title = exercise.upper()
    if rep_info:
        title += f"  |  Reps: {rep_info['rep_count']}"
    cv2.putText(panel, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # State indicator
    if rep_info:
        state_text = f"State: {rep_info['state']}  ({rep_info['tracked_angle']:.0f} deg)"
        cv2.putText(panel, state_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Feedback lines
    y = 72
    for text, severity in feedback_list:
        color = SEVERITY_COLORS.get(severity, (200, 200, 200))
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 20

    # Rep history (last 5 reps as colored dots with scores)
    if score_summary and score_summary["rep_history"]:
        y += 10
        cv2.putText(panel, "Rep History:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y += 20
        for rep in score_summary["rep_history"]:
            color = SCORE_COLORS.get(rep["score"], (150, 150, 150))
            # Colored dot
            cv2.circle(panel, (20, y - 5), 6, color, -1)
            text = f"  Rep {rep['rep_num']}: {rep['score']}  ({rep['min_angle']:.0f}-{rep['max_angle']:.0f} deg)"
            cv2.putText(panel, text, (32, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 18

    # Average score bar
    if score_summary and score_summary["total_reps"] > 0:
        y += 10
        avg = score_summary["avg_score"]
        bar_w = int((size[0] - 20) * avg)
        bar_color = (0, int(220 * avg), int(220 * (1 - avg)))
        cv2.rectangle(panel, (10, y), (10 + bar_w, y + 12), bar_color, -1)
        cv2.rectangle(panel, (10, y), (size[0] - 10, y + 12), (100, 100, 100), 1)
        pct = f"{avg * 100:.0f}% good"
        cv2.putText(panel, pct, (10, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return panel


def build_display(raw_frame, preprocessed, feature_grid, heatmap_grid,
                  skeleton_frame, feedback_list, exercise, angles,
                  canny_vis=None, skin_vis=None, sobel_vis=None,
                  rep_info=None, score_summary=None,
                  panel_w=320, panel_h=240):
    """Assemble 8 panels into a 2x4 display image.

    Layout:
      Row 1: Input Frame | Canny Edge Detection | HSV Skin Segmentation | Sobel Gradient
      Row 2: Heatmaps    | Skeleton Overlay      | Form + Rep Count      | CNN Features
    """

    def resize_panel(img, w, h):
        if img is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    def add_title(img, title):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(overlay, title, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        return overlay

    # Row 1: Input processing pipeline
    p1 = add_title(resize_panel(raw_frame, panel_w, panel_h), "1. Input Frame")
    p2 = add_title(resize_panel(canny_vis, panel_w, panel_h), "2. Canny Edge Detection")
    p3 = add_title(resize_panel(skin_vis, panel_w, panel_h), "3. HSV Skin Segmentation")
    p4 = add_title(resize_panel(sobel_vis, panel_w, panel_h), "4. Sobel Gradient Magnitude")

    # Row 2: Model output pipeline
    p5 = add_title(resize_panel(heatmap_grid, panel_w, panel_h), "5. Heatmaps (Spatial Probability)")
    p6 = add_title(resize_panel(skeleton_frame, panel_w, panel_h), "6. Skeleton Overlay")

    p7 = draw_feedback_panel((panel_w, panel_h), feedback_list, exercise, angles,
                             rep_info, score_summary)
    p7 = add_title(p7, "7. Form Analysis + Rep Count")

    p8 = add_title(resize_panel(feature_grid, panel_w, panel_h), "8. CNN Feature Activations")

    row1 = np.hstack([p1, p2, p3, p4])
    row2 = np.hstack([p5, p6, p7, p8])
    display = np.vstack([row1, row2])

    return display
