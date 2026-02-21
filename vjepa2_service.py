#!/usr/bin/env python3
"""
TEELA V-JEPA2 Sidecar Service
==============================
Runs V-JEPA2 (ViT-L) for motion/action prediction on live camera frames.
Publishes predictions to the kernel message bus.
Sends heartbeats so the kernel can auto-restart on crash.

Uses the smaller ViT-L (not ViT-G) to fit in Jetson Orin Nano 8GB
alongside SAM2. If RAM is tight, reduce NUM_FRAMES or resolution.
"""

import zmq
import json
import time
import threading
import cv2
import numpy as np
import torch
from collections import deque
from pathlib import Path

# ---------- CONFIG ----------
KERNEL_ROUTER = "tcp://localhost:7777"
KERNEL_PUB = "tcp://localhost:7778"
SERVICE_ID = "vjepa2"
HEARTBEAT_INTERVAL = 3
CAMERA_ID = 0                    # Left eye
NUM_FRAMES = 16                  # Temporal window (16 = good balance)
FRAME_SIZE = 256                 # V-JEPA2 native resolution
INFERENCE_INTERVAL = 2.0         # Seconds between predictions
MODEL_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"

class VJEPA2Service:
    def __init__(self):
        self.ctx = zmq.Context()
        self.running = False
        self.frame_buffer = deque(maxlen=NUM_FRAMES)
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_prediction = None
        self.cap = None

    def start(self):
        print(f"[vjepa2] Starting V-JEPA2 service on {self.device}")
        self.running = True

        # Load model
        try:
            from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification
            print(f"[vjepa2] Loading {MODEL_REPO}...")
            self.processor = AutoVideoProcessor.from_pretrained(MODEL_REPO)
            self.model = VJEPA2ForVideoClassification.from_pretrained(MODEL_REPO).to(self.device)
            self.model.eval()
            print(f"[vjepa2] Model loaded on {self.device}")
        except Exception as e:
            print(f"[vjepa2] FATAL: Cannot load model: {e}")
            raise

        # Open camera
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            gst = (f"v4l2src device=/dev/video{CAMERA_ID} ! "
                   f"video/x-raw,width={FRAME_SIZE},height={FRAME_SIZE} ! "
                   f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1")
            self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"[vjepa2] FATAL: Cannot open camera {CAMERA_ID}")
            raise RuntimeError(f"Camera {CAMERA_ID} unavailable")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ZMQ sockets
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, SERVICE_ID)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(KERNEL_ROUTER)

        # Start threads
        threading.Thread(target=self._heartbeat, daemon=True).start()
        threading.Thread(target=self._capture_loop, daemon=True).start()

        print(f"[vjepa2] Running (interval={INFERENCE_INTERVAL}s, frames={NUM_FRAMES})")
        self._inference_loop()

    def _capture_loop(self):
        """Continuously capture frames into the ring buffer."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                self.frame_buffer.append(rgb)
            time.sleep(1.0 / 15)  # ~15 fps capture

    def _inference_loop(self):
        """Run V-JEPA2 inference at regular intervals."""
        while self.running:
            try:
                if len(self.frame_buffer) >= NUM_FRAMES:
                    frames = np.array(list(self.frame_buffer))  # (T, H, W, 3)

                    inputs = self.processor(frames, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits

                    predicted_idx = logits.argmax(-1).item()
                    confidence = torch.softmax(logits, dim=-1).max().item()
                    label = self.model.config.id2label.get(predicted_idx, f"class_{predicted_idx}")

                    prediction = {
                        "type": "vjepa2_prediction",
                        "service_id": SERVICE_ID,
                        "action": label,
                        "action_id": predicted_idx,
                        "confidence": round(confidence, 3),
                        "num_frames": NUM_FRAMES,
                        "timestamp": time.time(),
                    }
                    self.last_prediction = prediction
                    self.dealer.send_json(prediction)

                time.sleep(INFERENCE_INTERVAL)

            except Exception as e:
                print(f"[vjepa2] Inference error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def _heartbeat(self):
        while self.running:
            try:
                self.dealer.send_json({
                    "type": "heartbeat",
                    "service_id": SERVICE_ID,
                    "last_action": self.last_prediction.get("action") if self.last_prediction else None,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            time.sleep(HEARTBEAT_INTERVAL)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.dealer.close()
        self.ctx.term()


if __name__ == "__main__":
    svc = VJEPA2Service()
    try:
        svc.start()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
