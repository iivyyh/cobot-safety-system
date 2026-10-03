import cv2
import math
import time
import csv
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading
import queue

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import winsound
except ImportError:
    winsound = None


class SafetyState(Enum):
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    STOP = "STOP"
    FAIL_SAFE = "FAIL_SAFE"


@dataclass
class SafetyConfig:
    cobot_origin: tuple = (430, 300)
    red_zone_r: int = 105
    warning_zone_r: int = 210
    min_contour_area: int = 1200
    camera_index: int = 0
    log_path: str = "safety_log.csv"
    morphology_kernel: int = 5
    trail_length: int = 30
    trend_window: int = 8
    panel_width: int = 300
    log_every_n_frames: int = 5
    hysteresis_px: int = 18
    state_confirm_frames: int = 3
    detection_hold_seconds: float = 2.0
    warmup_frames: int = 25
    audio_enabled: bool = True
    audio_language: str = "EN"  # EN or SR
    speech_rate: int = 165


class CobotSafetyMonitor:
    """Experimental vision-based cobot safety monitor.

    Educational prototype for demonstrating NORMAL/SLOW/STOP and fail-safe
    logic. It is not a safety-rated industrial controller.
    """

    def __init__(self, config: SafetyConfig):
        self.cfg = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=32, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morphology_kernel, self.cfg.morphology_kernel),
        )
        self.log_file = Path(self.cfg.log_path)
        self._init_log()

        self.primary_track = deque(maxlen=self.cfg.trail_length)
        self.distance_history = deque(maxlen=120)
        self.events = deque(maxlen=5)
        self.last_state = SafetyState.NORMAL
        self.warning_count = 0
        self.stop_count = 0
        self.fail_safe_count = 0
        self.frame_no = 0
        # Stabilisation / persistence state. MOG2 detects motion, so a person who
        # becomes still can temporarily disappear from the foreground mask.
        self.stable_state = SafetyState.NORMAL
        self.pending_state = SafetyState.NORMAL
        self.pending_count = 0
        self.last_detection_time = None
        self.last_known_distance = None
        self.last_known_center = None

        # Non-blocking audio alerts. Speech runs on a worker thread so camera
        # processing is not paused while a warning is being spoken.
        self.audio_enabled = self.cfg.audio_enabled
        self.audio_language = self.cfg.audio_language.upper()
        self.audio_queue = queue.Queue(maxsize=4)
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

    def _init_log(self):
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "state", "distance_px", "processing_ms", "fps"]
                )

    def calculate_distance(self, point):
        return math.hypot(
            point[0] - self.cfg.cobot_origin[0],
            point[1] - self.cfg.cobot_origin[1],
        )

    def evaluate_safety_state(self, distance):
        """Raw zone classification with hysteresis around zone boundaries."""
        h = self.cfg.hysteresis_px

        # Leaving a more dangerous state requires crossing a slightly wider
        # boundary. This prevents rapid STOP/SLOW/NORMAL flicker.
        if self.stable_state == SafetyState.STOP:
            if distance <= self.cfg.red_zone_r + h:
                return SafetyState.STOP
        elif self.stable_state == SafetyState.SLOW:
            if distance <= self.cfg.red_zone_r:
                return SafetyState.STOP
            if distance <= self.cfg.warning_zone_r + h:
                return SafetyState.SLOW

        if distance <= self.cfg.red_zone_r:
            return SafetyState.STOP
        if distance <= self.cfg.warning_zone_r:
            return SafetyState.SLOW
        return SafetyState.NORMAL

    def stabilise_state(self, candidate):
        """STOP is immediate; safer exits need several confirming frames."""
        if candidate == SafetyState.STOP:
            self.stable_state = SafetyState.STOP
            self.pending_state = candidate
            self.pending_count = 0
            return self.stable_state

        # Escalation from NORMAL to SLOW is also immediate.
        if (self.stable_state == SafetyState.NORMAL and
                candidate == SafetyState.SLOW):
            self.stable_state = SafetyState.SLOW
            self.pending_state = candidate
            self.pending_count = 0
            return self.stable_state

        if candidate == self.stable_state:
            self.pending_state = candidate
            self.pending_count = 0
            return self.stable_state

        if candidate != self.pending_state:
            self.pending_state = candidate
            self.pending_count = 1
        else:
            self.pending_count += 1

        if self.pending_count >= self.cfg.state_confirm_frames:
            self.stable_state = candidate
            self.pending_count = 0

        return self.stable_state

    @staticmethod
    def state_color(state):
        # OpenCV uses BGR. SLOW is orange, not yellow.
        return {
            SafetyState.NORMAL: (0, 210, 0),
            SafetyState.SLOW: (0, 165, 255),
            SafetyState.STOP: (0, 0, 255),
            SafetyState.FAIL_SAFE: (0, 0, 255),
        }[state]

    @staticmethod
    def safety_priority(state):
        return {
            SafetyState.NORMAL: 0,
            SafetyState.SLOW: 1,
            SafetyState.STOP: 2,
            SafetyState.FAIL_SAFE: 3,
        }[state]

    def detect_moving_objects(self, frame):
        mask = self.bg_subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.cfg.min_contour_area]
        return contours, mask

    def send_robot_command(self, state):
        return {
            SafetyState.NORMAL: "NORMAL SPEED",
            SafetyState.SLOW: "REDUCE SPEED",
            SafetyState.STOP: "STOP ROBOT",
            SafetyState.FAIL_SAFE: "FAIL-SAFE STOP",
        }[state]

    def log_measurement(self, state, distance, processing_ms, fps):
        with self.log_file.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                state.value,
                "" if distance is None else round(distance, 2),
                round(processing_ms, 3),
                round(fps, 2),
            ])

    def add_event(self, text):
        self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

    def register_transition(self, state):
        if state == self.last_state:
            return
        previous = self.last_state
        self.add_event(f"{previous.value} -> {state.value}")
        if state == SafetyState.SLOW:
            self.warning_count += 1
        elif state == SafetyState.STOP:
            self.stop_count += 1
        elif state == SafetyState.FAIL_SAFE:
            self.fail_safe_count += 1
        self.last_state = state
        self.queue_audio_alert(state)

    def _audio_worker(self):
        """Speak queued safety alerts without blocking the video loop."""
        engine = None
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.cfg.speech_rate)
            except Exception:
                engine = None

        while True:
            item = self.audio_queue.get()
            if item is None:
                break
            state, text = item
            if not self.audio_enabled:
                continue

            # Distinct short tones make the state recognisable even if TTS is
            # unavailable. On non-Windows systems the tone is simply skipped.
            if winsound is not None:
                try:
                    # Very simple rhythm-based signatures for a warehouse worker:
                    # NORMAL = ONE confirmation beep
                    # SLOW   = medium-paced warning beeps
                    # STOP   = rapid urgent beeps
                    # The rhythm, not only the pitch, identifies the safety state.
                    if state == SafetyState.NORMAL:
                        # SAFE: one calm confirmation sound.
                        winsound.Beep(650, 350)
                    elif state == SafetyState.SLOW:
                        # SLOW: clearly separated medium-speed beeps.
                        for i in range(4):
                            winsound.Beep(850, 180)
                            if i < 3:
                                time.sleep(0.32)
                    elif state == SafetyState.STOP:
                        # STOP: rapid alarm burst - intentionally much faster than SLOW.
                        for i in range(10):
                            winsound.Beep(1250, 85)
                            if i < 9:
                                time.sleep(0.045)
                    elif state == SafetyState.FAIL_SAFE:
                        # FAIL_SAFE: long emergency tone followed by a rapid burst.
                        winsound.Beep(1500, 700)
                        time.sleep(0.10)
                        for i in range(8):
                            winsound.Beep(1350, 90)
                            if i < 7:
                                time.sleep(0.05)
                except Exception:
                    pass

            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass

    def audio_message(self, state):
        messages = {
            "EN": {
                SafetyState.NORMAL: "Safe zone. Normal operation.",
                SafetyState.SLOW: "Warning. Safety zone entered. Reduce speed.",
                SafetyState.STOP: "Danger. Stop immediately.",
                SafetyState.FAIL_SAFE: "Safety system failure. Stop immediately.",
            },
            "SR": {
                SafetyState.NORMAL: "Bezbedna zona. Normalan rad.",
                SafetyState.SLOW: "Upozorenje. Ulazak u zonu upozorenja. Smanjite brzinu.",
                SafetyState.STOP: "Opasnost. Odmah zaustavite robota.",
                SafetyState.FAIL_SAFE: "Greska bezbednosnog sistema. Odmah zaustavite robota.",
            },
        }
        return messages.get(self.audio_language, messages["EN"])[state]

    def queue_audio_alert(self, state):
        if not self.audio_enabled:
            return
        # Alerts are generated only on a confirmed state transition, never on
        # every frame. Clear obsolete queued speech so the newest safety state
        # has priority.
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.audio_queue.put_nowait((state, self.audio_message(state)))
        except queue.Full:
            pass

    def motion_trend(self):
        if len(self.distance_history) < self.cfg.trend_window:
            return "STABLE", 0.0
        values = list(self.distance_history)[-self.cfg.trend_window:]
        delta = values[-1] - values[0]
        if delta < -12:
            return "APPROACHING", delta
        if delta > 12:
            return "MOVING AWAY", delta
        return "STABLE", delta

    def risk_level(self, distance, state):
        if state in (SafetyState.STOP, SafetyState.FAIL_SAFE):
            return "CRITICAL", 1.0
        if distance is None:
            return "LOW", 0.08
        max_r = float(self.cfg.warning_zone_r * 1.65)
        risk = 1.0 - min(max(distance - self.cfg.red_zone_r, 0.0) / max(max_r - self.cfg.red_zone_r, 1.0), 1.0)
        if state == SafetyState.SLOW:
            return "HIGH", max(risk, 0.62)
        if risk >= 0.42:
            return "MODERATE", risk
        return "LOW", risk

    def draw_transparent_zones(self, frame):
        overlay = frame.copy()
        center = self.cfg.cobot_origin
        cv2.circle(overlay, center, self.cfg.warning_zone_r, (0, 165, 255), -1)
        cv2.circle(overlay, center, self.cfg.red_zone_r, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.circle(frame, center, self.cfg.warning_zone_r, (0, 165, 255), 2)
        cv2.circle(frame, center, self.cfg.red_zone_r, (0, 0, 255), 2)

        # Stylized cobot base/arm instead of a plain origin point.
        x, y = center
        cv2.rectangle(frame, (x - 24, y + 18), (x + 24, y + 34), (210, 210, 210), -1)
        cv2.circle(frame, (x, y + 12), 13, (220, 220, 220), -1)
        cv2.line(frame, (x, y + 6), (x + 30, y - 28), (235, 235, 235), 8)
        cv2.circle(frame, (x + 30, y - 28), 8, (255, 255, 255), -1)
        cv2.line(frame, (x + 30, y - 28), (x + 52, y - 10), (235, 235, 235), 7)
        cv2.putText(frame, "COBOT", (x - 30, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "DANGER", (x - 33, max(20, y - self.cfg.red_zone_r - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
        cv2.putText(frame, "WARNING", (x - 40, max(20, y - self.cfg.warning_zone_r - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)

    def draw_trail(self, frame):
        pts = list(self.primary_track)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = max(1, int(4 * i / max(len(pts), 1)))
            cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), thickness)
        if len(pts) >= 2 and pts[-1] and pts[-2]:
            cv2.arrowedLine(frame, pts[-2], pts[-1], (255, 255, 255), 2, tipLength=0.35)

    def draw_cv_inset(self, frame, mask):
        h, w = frame.shape[:2]
        inset_w, inset_h = 190, 120
        small = cv2.resize(mask, (inset_w, inset_h))
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        x1, y1 = 12, h - inset_h - 34
        frame[y1:y1 + inset_h, x1:x1 + inset_w] = small
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x1 + inset_w + 2, y1 + inset_h + 2), (255, 255, 255), 1)
        cv2.putText(frame, "CV / MOTION DETECTION", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    def draw_distance_graph(self, panel, x, y, width, height):
        cv2.rectangle(panel, (x, y), (x + width, y + height), (55, 55, 55), 1)
        cv2.putText(panel, "DISTANCE HISTORY", (x + 8, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
        vals = [v for v in self.distance_history if v is not None]
        if len(vals) < 2:
            return
        vals = vals[-60:]
        max_v = max(max(vals), self.cfg.warning_zone_r * 1.3, 1)
        points = []
        for i, val in enumerate(vals):
            px = x + 5 + int(i * (width - 10) / max(len(vals) - 1, 1))
            py = y + height - 5 - int(min(val / max_v, 1.0) * (height - 28))
            points.append((px, py))
        for i in range(1, len(points)):
            cv2.line(panel, points[i - 1], points[i], (220, 220, 220), 1)

    def build_dashboard(self, frame, state, distance, fps, processing_ms, objects, trend, risk_name, risk_value, command, mask):
        h, w = frame.shape[:2]
        panel = cv2.copyMakeBorder(frame, 0, 0, 0, self.cfg.panel_width, cv2.BORDER_CONSTANT, value=(18, 18, 18))
        px = w + 18
        color = self.state_color(state)

        cv2.putText(panel, "COBOT SAFETY", (px, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
        cv2.putText(panel, "MONITOR", (px, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
        cv2.putText(panel, "Q quit | R reset | A audio | L language", (px, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (125, 125, 125), 1)
        cv2.line(panel, (px, 70), (w + self.cfg.panel_width - 18, 70), (70, 70, 70), 1)

        cv2.putText(panel, "SYSTEM STATE", (px, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(panel, state.value, (px, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.86, color, 3)

        rows = [
            ("DISTANCE", "--" if distance is None else f"{distance:.0f} px"),
            ("TREND", trend),
            ("RISK", risk_name),
            ("OBJECTS", str(objects)),
            ("FPS", f"{fps:.1f}"),
            ("LATENCY", f"{processing_ms:.1f} ms"),
            ("CAMERA", "ONLINE"),
            ("LOGGING", "ACTIVE"),
            ("AUDIO", "ON" if self.audio_enabled else "OFF"),
            ("VOICE", self.audio_language),
        ]
        yy = 158
        for label, value in rows:
            cv2.putText(panel, label, (px, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (145, 145, 145), 1)
            cv2.putText(panel, value, (px + 112, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1)
            yy += 24

        # Risk bar
        bar_y = yy + 4
        bar_w = self.cfg.panel_width - 42
        cv2.rectangle(panel, (px, bar_y), (px + bar_w, bar_y + 12), (65, 65, 65), -1)
        cv2.rectangle(panel, (px, bar_y), (px + int(bar_w * max(0.0, min(risk_value, 1.0))), bar_y + 12), color, -1)

        cmd_y = bar_y + 42
        cv2.putText(panel, "ROBOT COMMAND", (px, cmd_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(panel, command, (px, cmd_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        graph_y = cmd_y + 45
        graph_h = 100
        if graph_y + graph_h < h - 100:
            self.draw_distance_graph(panel, px, graph_y, self.cfg.panel_width - 38, graph_h)

        # Event footer
        event_y = max(graph_y + graph_h + 24, h - 86)
        cv2.putText(panel, "EVENT HISTORY", (px, event_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        for i, event in enumerate(list(self.events)[:3]):
            cv2.putText(panel, event, (px, event_y + 18 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)

        self.draw_cv_inset(panel[:, :w], mask)
        return panel

    def draw_alert_banner(self, frame, state, trend):
        h, w = frame.shape[:2]
        if state == SafetyState.NORMAL:
            text = "WORKSPACE SAFE"
        elif state == SafetyState.SLOW:
            text = "WARNING - OPERATOR APPROACHING" if trend == "APPROACHING" else "WARNING - SAFETY ZONE ENTERED"
        elif state == SafetyState.STOP:
            text = "CRITICAL PROXIMITY - STOP"
        else:
            text = "CAMERA LOST - FAIL-SAFE STOP"
        color = self.state_color(state)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        cv2.putText(frame, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    def run(self):
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Camera is unavailable. FAIL_SAFE / STOP required.")

        previous_time = time.perf_counter()
        self.add_event("SYSTEM STARTED")

        try:
            while True:
                frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.register_transition(SafetyState.FAIL_SAFE)
                    self.fail_safe_count += 1
                    print("[FAIL-SAFE] Video stream lost -> STOP.")
                    break

                # Keep the original camera resolution, but move the virtual cobot
                # into frame if a smaller camera resolution is used.
                h, w = frame.shape[:2]
                ox = min(max(self.cfg.cobot_origin[0], 80), w - 80)
                oy = min(max(self.cfg.cobot_origin[1], 80), h - 80)
                self.cfg.cobot_origin = (ox, oy)

                contours, mask = self.detect_moving_objects(frame)
                # Give MOG2 a short period to learn the initial background.
                if self.frame_no < self.cfg.warmup_frames:
                    contours = []
                detections = []
                for contour in contours:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    center = (int(x + bw / 2), int(y + bh / 2))
                    distance = self.calculate_distance(center)
                    state = self.evaluate_safety_state(distance)
                    detections.append((state, distance, (x, y, bw, bh), center))

                now_monotonic = time.perf_counter()
                if detections:
                    selected = max(detections, key=lambda d: self.safety_priority(d[0]))
                    candidate_state, nearest_distance, _, primary_center = selected
                    self.last_detection_time = now_monotonic
                    self.last_known_distance = nearest_distance
                    self.last_known_center = primary_center
                    self.primary_track.append(primary_center)
                    self.distance_history.append(nearest_distance)
                else:
                    # MOG2 can absorb a stationary person into the background.
                    # Never reinterpret disappearance from a hazardous zone as SAFE.
                    elapsed = (now_monotonic - self.last_detection_time
                               if self.last_detection_time is not None else None)
                    if self.stable_state in (SafetyState.SLOW, SafetyState.STOP):
                        candidate_state = self.stable_state
                        nearest_distance = self.last_known_distance
                        if self.last_known_center is not None:
                            self.primary_track.append(self.last_known_center)
                            if nearest_distance is not None:
                                self.distance_history.append(nearest_distance)
                    elif elapsed is not None and elapsed <= self.cfg.detection_hold_seconds:
                        candidate_state = self.stable_state
                        nearest_distance = self.last_known_distance
                        self.primary_track.append(self.last_known_center)
                    else:
                        candidate_state = SafetyState.NORMAL
                        nearest_distance = None
                        self.primary_track.append(None)

                current_state = self.stabilise_state(candidate_state)
                previous_logged_state = self.last_state
                self.register_transition(current_state)
                trend, _ = self.motion_trend()
                risk_name, risk_value = self.risk_level(nearest_distance, current_state)
                command = self.send_robot_command(current_state)

                self.draw_transparent_zones(frame)

                for idx, (state, distance, (x, y, bw, bh), center) in enumerate(detections, start=1):
                    color = self.state_color(state)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                    cv2.circle(frame, center, 4, color, -1)
                    cv2.line(frame, center, self.cfg.cobot_origin, color, 1)
                    cv2.putText(frame, f"OBJECT #{idx} | {distance:.0f}px | {state.value}",
                                (x, max(54, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

                self.draw_trail(frame)
                self.draw_alert_banner(frame, current_state, trend)

                now = time.perf_counter()
                dt = max(now - previous_time, 1e-9)
                fps = 1.0 / dt
                previous_time = now
                processing_ms = (time.perf_counter() - frame_start) * 1000.0

                self.frame_no += 1
                if (self.frame_no % self.cfg.log_every_n_frames == 0 or
                        current_state != previous_logged_state):
                    self.log_measurement(current_state, nearest_distance, processing_ms, fps)

                dashboard = self.build_dashboard(
                    frame, current_state, nearest_distance, fps, processing_ms,
                    len(detections), trend, risk_name, risk_value, command, mask
                )

                cv2.imshow("Cobot Safety Monitoring Dashboard", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("a"):
                    self.audio_enabled = not self.audio_enabled
                    self.add_event(f"AUDIO {'ON' if self.audio_enabled else 'OFF'}")
                elif key == ord("l"):
                    self.audio_language = "SR" if self.audio_language == "EN" else "EN"
                    self.add_event(f"VOICE LANGUAGE -> {self.audio_language}")
                    self.queue_audio_alert(current_state)
                elif key == ord("r"):
                    # Manual demo reset after the workspace has been visually checked.
                    self.stable_state = SafetyState.NORMAL
                    self.pending_state = SafetyState.NORMAL
                    self.pending_count = 0
                    self.last_detection_time = None
                    self.last_known_distance = None
                    self.last_known_center = None
                    self.primary_track.clear()
                    self.distance_history.clear()
                    self.add_event("MANUAL RESET -> NORMAL")
                    self.last_state = SafetyState.NORMAL
                    self.queue_audio_alert(SafetyState.NORMAL)
                elif key == ord("q") or key == 27:
                    self.add_event("SYSTEM STOPPED BY USER")
                    break

        finally:
            try:
                self.audio_queue.put_nowait(None)
            except queue.Full:
                pass
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    CobotSafetyMonitor(SafetyConfig()).run()
