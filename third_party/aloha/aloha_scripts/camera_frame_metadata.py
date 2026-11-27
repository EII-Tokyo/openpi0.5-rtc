"""Read-only frame telemetry; errors must never stop camera image publication."""
import json
import math
import time


def frame_exposure(frame, metadata, received_unix):
    result = {"received_unix": received_unix, "frame_number": None,
              "actual_exposure_ms": None, "gain_level": None, "auto_exposure": None,
              "source": "frame_metadata", "error": None}
    try:
        result["frame_number"] = frame.get_frame_number()
        for name, key, scale in [("actual_exposure", "actual_exposure_ms", 0.001),
                                 ("gain_level", "gain_level", 1),
                                 ("auto_exposure", "auto_exposure", 1)]:
            option = getattr(metadata, name)
            if frame.supports_frame_metadata(option):
                value = float(frame.get_frame_metadata(option))
                if math.isfinite(value) and value >= 0:
                    result[key] = bool(value) if key == "auto_exposure" else value * scale
        if result["actual_exposure_ms"] is None:
            result["error"] = "actual_exposure metadata unsupported"
    except Exception as exc:
        result["error"] = str(exc)[:160]
    return result


class ExposureMetadataPublisher:
    """Inspect every delivered frame; send only the latest sample at most 5 Hz."""
    def __init__(self, publisher, metadata, clock=time.monotonic):
        self.publisher = publisher
        self.metadata = metadata
        self.clock = clock
        self.latest = {}
        self.last_publish = float("-inf")

    def capture(self, name, frame, received_unix):
        self.latest[name] = frame_exposure(frame, self.metadata, received_unix)

    def publish_latest(self):
        now = self.clock()
        if now - self.last_publish < 0.2:
            return
        self.last_publish = now
        try:
            self.publisher.publish(json.dumps({"version": 1, "cameras": self.latest}, allow_nan=False))
        except Exception:
            # Telemetry is optional and never blocks/fails the image path.
            pass
