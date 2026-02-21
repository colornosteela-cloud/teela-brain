#!/usr/bin/env python3
"""
TEELA SAM2 Sidecar Service
============================
Runs SAM2 (Segment Anything Model 2) for real-time object segmentation.
Publishes segmentation masks + object info to kernel message bus.
Uses SAM2-tiny or SAM2-small to fit in Jetson 8GB alongside V-JEPA2.
"""

import zmq
import json
import time
import threading
import cv2
import numpy as np
import torch
from pathlib import Path

# ---------- CONFIG ----------
KERNEL_ROUTER = "tcp://localhost:7777"
SERVICE_ID = "sam2"
HEARTBEAT_INTERVAL = 3
CAMERA_ID = 0
INFERENCE_INTERVAL = 1.5         # Segment every 1.5s
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# SAM2 model choice - use tiny for Jetson Orin Nano
SAM2_CHECKPOINT = "facebook/sam2.1-hiera-tiny"

class SAM2Service:
    def __init__(self):
        self.ctx = zmq.Context()
        self.running = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.cap = None
        self.last_result = None

    def start(self):
        print(f"[sam2] Starting SAM2 service on {self.device}")
        self.running = True

        # Load SAM2
        try:
            from transformers import AutoProcessor, AutoModelForMaskGeneration
            print(f"[sam2] Loading {SAM2_CHECKPOINT}...")
            self.processor = AutoProcessor.from_pretrained(SAM2_CHECKPOINT)
            self.model = AutoModelForMaskGeneration.from_pretrained(SAM2_CHECKPOINT).to(self.device)
            self.model.eval()
            print(f"[sam2] Model loaded on {self.device}")
        except ImportError:
            # Fallback: use sam2 package directly
            try:
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                print(f"[sam2] Using sam2 package...")
                sam2 = build_sam2("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt", device=self.device)
                self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
                self.model = "sam2_native"
                print(f"[sam2] SAM2 native loaded")
            except Exception as e2:
                print(f"[sam2] FATAL: Cannot load SAM2: {e2}")
                raise

        # Open camera
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            gst = (f"v4l2src device=/dev/video{CAMERA_ID} ! "
                   f"video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT} ! "
                   f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1")
            self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {CAMERA_ID} unavailable")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ZMQ
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, SERVICE_ID)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(KERNEL_ROUTER)

        threading.Thread(target=self._heartbeat, daemon=True).start()

        print(f"[sam2] Running (interval={INFERENCE_INTERVAL}s)")
        self._inference_loop()

    def _inference_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t0 = time.time()

                if self.model == "sam2_native":
                    masks = self.mask_generator.generate(rgb)
                    objects = []
                    for i, m in enumerate(masks[:10]):  # Top 10 segments
                        bbox = m["bbox"]  # [x, y, w, h]
                        objects.append({
                            "id": i,
                            "area": int(m["area"]),
                            "bbox": [int(x) for x in bbox],
                            "stability": round(float(m["stability_score"]), 3),
                            "center": [int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)],
                        })
                else:
                    # HuggingFace transformers path
                    inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    # Extract masks and compute object info
                    masks_tensor = outputs.pred_masks[0] if hasattr(outputs, 'pred_masks') else None
                    objects = []
                    if masks_tensor is not None:
                        for i in range(min(10, masks_tensor.shape[0])):
                            mask = masks_tensor[i].cpu().numpy() > 0
                            area = int(mask.sum())
                            if area < 100:
                                continue
                            ys, xs = np.where(mask)
                            objects.append({
                                "id": i,
                                "area": area,
                                "bbox": [int(xs.min()), int(ys.min()),
                                         int(xs.max()-xs.min()), int(ys.max()-ys.min())],
                                "center": [int(xs.mean()), int(ys.mean())],
                            })

                elapsed = time.time() - t0
                result = {
                    "type": "sam2_segmentation",
                    "service_id": SERVICE_ID,
                    "num_objects": len(objects),
                    "objects": objects,
                    "inference_ms": round(elapsed * 1000, 1),
                    "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
                    "timestamp": time.time(),
                }
                self.last_result = result
                self.dealer.send_json(result)

                time.sleep(max(0, INFERENCE_INTERVAL - elapsed))

            except Exception as e:
                print(f"[sam2] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def _heartbeat(self):
        while self.running:
            try:
                self.dealer.send_json({
                    "type": "heartbeat",
                    "service_id": SERVICE_ID,
                    "num_objects": self.last_result.get("num_objects") if self.last_result else 0,
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
    svc = SAM2Service()
    try:
        svc.start()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
