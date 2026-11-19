"""Home-return gate for one commanded bottle cycle (not a task-success detector)."""
import numpy as np


class HomeReturnGate:
    # Grippers use normalized units and are deliberately excluded from arm tolerances.
    ARM_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])

    def __init__(self, position_tolerance=0.05, departure_tolerance=0.15,
                 velocity_tolerance=0.1, stable_seconds=0.3):
        values = np.array([position_tolerance, departure_tolerance, velocity_tolerance, stable_seconds])
        if not np.isfinite(values).all() or np.any(values <= 0) or departure_tolerance <= position_tolerance:
            raise ValueError("Home thresholds must be positive; departure must exceed arrival tolerance")
        self.position_tolerance = position_tolerance
        self.departure_tolerance = departure_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.stable_seconds = stable_seconds
        self.home = None
        self.reset()

    @staticmethod
    def vector(value):
        result = np.asarray(value, dtype=float)
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError("Home detection requires 14 finite measured joint values")
        return result

    def set_home(self, qpos):
        self.home = self.vector(qpos).copy()
        self.reset()

    def reset(self):
        self.departed = False
        self.settling = False
        self.stable_since = None

    def at_home(self, qpos):
        if self.home is None:
            raise RuntimeError("Home reference has not been captured after environment reset")
        error = np.max(np.abs((self.vector(qpos) - self.home)[self.ARM_INDICES]))
        return error <= self.position_tolerance

    def update(self, qpos, qvel, now):
        qpos, qvel = self.vector(qpos), self.vector(qvel)
        near = self.at_home(qpos)
        if not self.departed:
            self.departed = np.max(np.abs((qpos - self.home)[self.ARM_INDICES])) > self.departure_tolerance
            return "running"
        slow = np.max(np.abs(qvel[self.ARM_INDICES])) <= self.velocity_tolerance
        # A fast pass through home is not completion. Once settled, stop policy playback.
        if near and slow:
            self.settling = True
            if self.stable_since is None:
                self.stable_since = now
            if now - self.stable_since >= self.stable_seconds:
                return "complete"
        else:
            self.stable_since = None
        return "settling" if self.settling else "running"

    def hold_action(self, qpos):
        action = self.vector(qpos).copy()
        action[self.ARM_INDICES] = self.home[self.ARM_INDICES]
        return {"actions": action}
