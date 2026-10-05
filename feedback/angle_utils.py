"""Joint angle computation from 2D keypoint coordinates.

Basic 2D geometry: given three points (p1, vertex, p3),
compute the angle at the vertex using atan2.
"""

import numpy as np


def compute_angle(p1, p2, p3):
    """Compute angle at vertex p2 formed by rays p2->p1 and p2->p3.

    Args:
        p1, p2, p3: (x, y) coordinates as arrays or tuples

    Returns:
        Angle in degrees [0, 180]
    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


# COCO keypoint indices for convenience
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def get_joint_angles(keypoints):
    """Compute key joint angles from 17 keypoints.

    Args:
        keypoints: (17, 2) array of (x, y) coordinates

    Returns:
        Dict of angle_name -> degrees
    """
    kp = keypoints

    angles = {}

    # Knee angles (hip-knee-ankle)
    angles["left_knee"] = compute_angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
    angles["right_knee"] = compute_angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])

    # Elbow angles (shoulder-elbow-wrist)
    angles["left_elbow"] = compute_angle(kp[L_SHOULDER], kp[L_ELBOW], kp[L_WRIST])
    angles["right_elbow"] = compute_angle(kp[R_SHOULDER], kp[R_ELBOW], kp[R_WRIST])

    # Hip angles (shoulder-hip-knee)
    angles["left_hip"] = compute_angle(kp[L_SHOULDER], kp[L_HIP], kp[L_KNEE])
    angles["right_hip"] = compute_angle(kp[R_SHOULDER], kp[R_HIP], kp[R_KNEE])

    # Back angle (midpoint shoulders - midpoint hips - midpoint knees)
    mid_shoulder = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
    mid_hip = (kp[L_HIP] + kp[R_HIP]) / 2
    mid_knee = (kp[L_KNEE] + kp[R_KNEE]) / 2
    angles["back"] = compute_angle(mid_shoulder, mid_hip, mid_knee)

    return angles
