#!/usr/bin/env python3
"""
TEELA Camera Service (Mock)
============================
Mock camera service for testing when hardware cameras aren't available.
Generates synthetic frames with moving shapes so other services
(V-JEPA2, SAM2, brain) have something to process.

When real cameras work, replace with camera_service.py that does
actual V4L2/GStreamer capture and publishes frames via shared memory
or ZMQ. The mock sends heartbeats identically to the real service
so the kernel treats it the same way.
"""

import zmq
import json
import time
import threading
import numpy as np
import cv2
import base64
from pathlib import Path

# ---------- CONFIG ----------
KERNEL_ROUTER = "tcp://localhost:7777"
SERVICE_ID = "camera"
HEARTBEAT_INTERVAL = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
PUBLISH_FRAMES = True  # Set True to publish base64 frames over ZMQ


class CameraServiceMock:
    def __init__(self):
        self.ctx = zmq.Context()
        self.running = False
        self.frame_count = 0
        self.dealer = None

    def start(self):
        print(f"[camera] Mock camera service starting")
        print(f"[camera] Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
        self.running = True

        # Connect to kernel
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, SERVICE_ID)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(KERNEL_ROUTER)

        # Start heartbeat
        threading.Thread(target=self._heartbeat, daemon=True).start()

        print(f"[camera] Mock camera running")
        self._frame_loop()

    def _frame_loop(self):
        """Generate and optionally publish synthetic frames."""
        while self.running:
            t0 = time.time()
            frame = self._generate_frame()
            self.frame_count += 1

            if PUBLISH_FRAMES and self.frame_count % 5 == 0:
                # Publish every 5th frame to reduce ZMQ load
                _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(jpg.tobytes()).decode()
                try:
                    self.dealer.send_json({
                        "type": "camera_frame",
                        "service_id": SERVICE_ID,
                        "camera": "left",
                        "width": FRAME_WIDTH,
                        "height": FRAME_HEIGHT,
                        "frame_num": self.frame_count,
                        "base64": b64,
                        "timestamp": time.time(),
                    })
                except zmq.ZMQError:
                    pass

            # Pace to target FPS
            elapsed = time.time() - t0
            sleep = max(0, (1.0 / FPS) - elapsed)
            time.sleep(sleep)

    def _generate_frame(self):
        """Create a synthetic frame with moving colored shapes."""
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Dark background with subtle grid
        for x in range(0, FRAME_WIDTH, 40):
            cv2.line(frame, (x, 0), (x, FRAME_HEIGHT), (20, 20, 30), 1)
        for y in range(0, FRAME_HEIGHT, 40):
            cv2.line(frame, (0, y), (FRAME_WIDTH, y), (20, 20, 30), 1)

        t = time.time()

        # Moving circle (simulates a person/object)
        cx = int(FRAME_WIDTH / 2 + 150 * np.sin(t * 0.5))
        cy = int(FRAME_HEIGHT / 2 + 80 * np.cos(t * 0.3))
        cv2.circle(frame, (cx, cy), 40, (0, 200, 255), -1)
        cv2.circle(frame, (cx, cy), 42, (0, 150, 200), 2)

        # Moving rectangle (simulates an obstacle)
        rx = int(100 + 80 * np.sin(t * 0.7 + 1.5))
        ry = int(300 + 50 * np.cos(t * 0.4))
        cv2.rectangle(frame, (rx, ry), (rx + 80, ry + 60), (255, 100, 0), -1)
        cv2.rectangle(frame, (rx, ry), (rx + 80, ry + 60), (200, 80, 0), 2)

        # Small moving dots (simulates particles/small objects)
        for i in range(5):
            dx = int(FRAME_WIDTH * (0.1 + 0.8 * ((t * 0.2 + i * 0.7) % 1.0)))
            dy = int(FRAME_HEIGHT * (0.1 + 0.8 * ((t * 0.15 + i * 0.5) % 1.0)))
            color = [(100, 255, 100), (255, 100, 100), (100, 100, 255),
                     (255, 255, 100), (255, 100, 255)][i]
            cv2.circle(frame, (dx, dy), 8, color, -1)

        # Status text overlay
        cv2.putText(frame, "TEELA MOCK CAMERA", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, f"{FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Simulated timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (FRAME_WIDTH - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return frame

    def _heartbeat(self):
        while self.running:
            try:
                self.dealer.send_json({
                    "type": "heartbeat",
                    "service_id": SERVICE_ID,
                    "frame_count": self.frame_count,
                    "mock": True,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            time.sleep(HEARTBEAT_INTERVAL)

    def stop(self):
        self.running = False
        if self.dealer:
            self.dealer.close()
        self.ctx.term()
        print(f"[camera] Mock camera stopped")


if __name__ == "__main__":
    svc = CameraServiceMock()
    try:
        svc.start()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
