"""Rule-based exercise form checker.

Given joint angles, check against thresholds and return
feedback strings with severity levels (good/warning/bad).
"""

from .angle_utils import get_joint_angles


def check_squat(keypoints):
    """Check squat form. Key checks: knee depth, back angle."""
    angles = get_joint_angles(keypoints)
    feedback = []

    # Average knee angle
    knee_avg = (angles["left_knee"] + angles["right_knee"]) / 2

    if knee_avg > 160:
        feedback.append(("Standing — start your squat", "info"))
    elif knee_avg > 120:
        feedback.append(("Go deeper — thighs not parallel yet", "warning"))
    elif knee_avg > 80:
        feedback.append(("Good depth", "good"))
    else:
        feedback.append(("Very deep squat — watch your knees", "warning"))

    # Back angle — should stay relatively upright
    if angles["back"] < 60:
        feedback.append(("Keep your chest up — too much forward lean", "bad"))
    elif angles["back"] < 80:
        feedback.append(("Slight forward lean — try to stay more upright", "warning"))
    else:
        feedback.append(("Good back position", "good"))

    return feedback


def check_pushup(keypoints):
    """Check push-up form. Key checks: elbow angle, hip sag."""
    angles = get_joint_angles(keypoints)
    feedback = []

    # Average elbow angle
    elbow_avg = (angles["left_elbow"] + angles["right_elbow"]) / 2

    if elbow_avg > 150:
        feedback.append(("Arms extended — lower yourself", "info"))
    elif elbow_avg > 110:
        feedback.append(("Go lower — elbows should reach ~90 degrees", "warning"))
    elif 70 <= elbow_avg <= 110:
        feedback.append(("Good depth", "good"))
    else:
        feedback.append(("Too low — push back up", "warning"))

    # Hip sag — hip angle should be roughly straight (160+)
    hip_avg = (angles["left_hip"] + angles["right_hip"]) / 2
    if hip_avg < 150:
        feedback.append(("Hips sagging — engage your core", "bad"))
    elif hip_avg < 165:
        feedback.append(("Keep your body straighter", "warning"))
    else:
        feedback.append(("Good body alignment", "good"))

    return feedback


def check_deadlift(keypoints):
    """Check deadlift form. Key checks: back rounding, lockout."""
    angles = get_joint_angles(keypoints)
    feedback = []

    # Back angle
    if angles["back"] < 70:
        feedback.append(("Back is rounding — straighten up", "bad"))
    elif angles["back"] < 90:
        feedback.append(("Watch your back — keep it neutral", "warning"))
    else:
        feedback.append(("Good back position", "good"))

    # Hip hinge
    hip_avg = (angles["left_hip"] + angles["right_hip"]) / 2
    if hip_avg > 160:
        feedback.append(("Lockout — stand tall", "good"))
    elif hip_avg > 120:
        feedback.append(("Drive your hips forward", "info"))
    else:
        feedback.append(("Good hinge position", "good"))

    # Knee
    knee_avg = (angles["left_knee"] + angles["right_knee"]) / 2
    if knee_avg < 100:
        feedback.append(("Don't squat the deadlift — less knee bend", "warning"))

    return feedback


EXERCISE_CHECKERS = {
    "squat": check_squat,
    "pushup": check_pushup,
    "deadlift": check_deadlift,
}


def check_form(keypoints, exercise):
    """Check exercise form and return list of (feedback, severity)."""
    checker = EXERCISE_CHECKERS.get(exercise)
    if checker is None:
        return [("Unknown exercise", "info")]
    return checker(keypoints)
