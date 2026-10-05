"""Per-rep form scoring.

Tracks form quality during each rep by accumulating feedback severities
from check_form(). Scores each completed rep as good/ok/bad and maintains
a history of recent rep scores.
"""


class RepScorer:

    def __init__(self, exercise="squat"):
        self.exercise = exercise
        self.rep_history = []  # List of {"rep_num", "score", "min_angle", "max_angle"}
        self.total_reps = 0
        self.good_reps = 0
        self.ok_reps = 0
        self.bad_reps = 0

        # Per-rep accumulator (reset each rep)
        self._collecting = False
        self._frame_count = 0
        self._bad_count = 0
        self._warning_count = 0
        self._min_angle = 180
        self._max_angle = 0

    def update(self, angles, feedback_list, rep_state):
        """Process one frame during a rep.

        Args:
            angles: dict from get_joint_angles()
            feedback_list: list of (text, severity) from check_form()
            rep_state: dict from RepCounter.update()
        """
        state = rep_state["state"]
        tracked = rep_state["tracked_angle"]

        # Start collecting when we leave standing
        if state in ("descending", "bottom", "ascending"):
            self._collecting = True
            self._frame_count += 1
            self._min_angle = min(self._min_angle, tracked)
            self._max_angle = max(self._max_angle, tracked)

            for _, severity in feedback_list:
                if severity == "bad":
                    self._bad_count += 1
                elif severity == "warning":
                    self._warning_count += 1

        # Rep just completed — finalize score
        if rep_state["just_completed"] and self._collecting:
            self.total_reps += 1

            if self._bad_count > 0:
                score = "bad"
                self.bad_reps += 1
            elif self._frame_count > 0 and self._warning_count / self._frame_count > 0.3:
                score = "ok"
                self.ok_reps += 1
            else:
                score = "good"
                self.good_reps += 1

            self.rep_history.append({
                "rep_num": self.total_reps,
                "score": score,
                "min_angle": self._min_angle,
                "max_angle": self._max_angle,
            })

            # Keep only last 5
            if len(self.rep_history) > 5:
                self.rep_history = self.rep_history[-5:]

            # Reset accumulator
            self._reset_accumulator()

        # Reset accumulator if we're back to standing without completing
        if state == "standing" and not rep_state["just_completed"]:
            self._reset_accumulator()

    def _reset_accumulator(self):
        self._collecting = False
        self._frame_count = 0
        self._bad_count = 0
        self._warning_count = 0
        self._min_angle = 180
        self._max_angle = 0

    def get_summary(self):
        total = self.total_reps or 1  # Avoid division by zero
        return {
            "total_reps": self.total_reps,
            "good_reps": self.good_reps,
            "ok_reps": self.ok_reps,
            "bad_reps": self.bad_reps,
            "avg_score": self.good_reps / total,
            "rep_history": list(self.rep_history),
        }

    def reset(self):
        self.rep_history = []
        self.total_reps = 0
        self.good_reps = 0
        self.ok_reps = 0
        self.bad_reps = 0
        self._reset_accumulator()
