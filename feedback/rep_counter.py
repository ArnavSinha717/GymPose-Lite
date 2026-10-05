"""Rep counter using state machine on joint angle time series.

Tracks the primary angle for each exercise (knee for squats, hip for
deadlifts, elbow for pushups) and counts completed reps by detecting
the full descent-ascent cycle.

Uses hysteresis to prevent jitter-triggered false transitions and
a moving average to smooth noisy angle readings.
"""

from collections import deque


# Exercise -> (angle keys to average, standing threshold, bottom threshold)
EXERCISE_CONFIG = {
    "squat": {
        "angle_keys": ["left_knee", "right_knee"],
        "standing": 150,
        "bottom": 110,
        "hysteresis": 8,
    },
    "deadlift": {
        "angle_keys": ["left_hip", "right_hip"],
        "standing": 150,
        "bottom": 110,
        "hysteresis": 8,
    },
    "pushup": {
        "angle_keys": ["left_elbow", "right_elbow"],
        "standing": 150,
        "bottom": 100,
        "hysteresis": 8,
    },
}


class RepCounter:

    def __init__(self, exercise="squat"):
        self.exercise = exercise
        self.config = EXERCISE_CONFIG.get(exercise, EXERCISE_CONFIG["squat"])
        self.state = "standing"
        self.rep_count = 0
        self.angle_history = deque(maxlen=10)

    def _get_tracked_angle(self, angles):
        """Average the relevant angles for this exercise."""
        keys = self.config["angle_keys"]
        values = [angles.get(k, 180) for k in keys]
        return sum(values) / len(values)

    def _smoothed_angle(self):
        """Moving average of recent angle values."""
        if not self.angle_history:
            return 180
        return sum(self.angle_history) / len(self.angle_history)

    def update(self, angles):
        """Process one frame's angles and update rep state.

        Args:
            angles: dict from get_joint_angles()

        Returns:
            dict with rep_count, state, tracked_angle, just_completed
        """
        raw_angle = self._get_tracked_angle(angles)
        self.angle_history.append(raw_angle)
        angle = self._smoothed_angle()

        standing = self.config["standing"]
        bottom = self.config["bottom"]
        hyst = self.config["hysteresis"]
        just_completed = False

        if self.state == "standing":
            if angle < standing - hyst:
                self.state = "descending"

        elif self.state == "descending":
            if angle < bottom:
                self.state = "bottom"
            elif angle > standing:
                self.state = "standing"  # Aborted rep

        elif self.state == "bottom":
            if angle > bottom + hyst:
                self.state = "ascending"

        elif self.state == "ascending":
            if angle > standing:
                self.state = "standing"
                self.rep_count += 1
                just_completed = True
            elif angle < bottom:
                self.state = "bottom"  # Went back down

        return {
            "rep_count": self.rep_count,
            "state": self.state,
            "tracked_angle": angle,
            "just_completed": just_completed,
        }

    def reset(self):
        self.state = "standing"
        self.rep_count = 0
        self.angle_history.clear()
