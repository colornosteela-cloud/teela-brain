#!/usr/bin/env python3
"""
TEELA CortexOS - Live Brain with V-JEPA2 + SAM2 Sidecars
==========================================================
Jetson Orin Nano. Ollama qwen3.5:397b-cloud.
Consumes V-JEPA2 (motion) + SAM2 (segmentation) sidecar data.
Separate conversation thread for continuous user interaction.
Web dashboard with live video at http://<jetson-ip>:8080

Architecture:
  [Camera] --> [V-JEPA2 svc] --> motion predictions --\
           --> [SAM2 svc]   --> object segments     --+--> [Frontal Lobe] --> [Motor]
           --> [Brain]      --> raw frames + VLM    --/       |
                                                              v
  [Web Dashboard :8080] <-- MJPEG streams + brain status + controls
  [Conversation Thread] <-- Ollama chat (non-blocking, always listening)
  [OpenClaw Skill]      <-- any chat channel
"""

import cv2, zmq, json, time, base64, threading, requests, re
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class RobotConfig:
    camera_left: int = 0
    camera_right: int = 2
    frame_width: int = 640
    frame_height: int = 480
    capture_fps: int = 30
    analysis_interval: float = 2.0
    vlm_width: int = 320
    vlm_height: int = 240
    ollama_url: str = "http://localhost:11434"
    vlm_model: str = "qwen3.5:397b-cloud"
    vlm_timeout: int = 45
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    kernel_router: str = "tcp://localhost:7777"
    kernel_pub: str = "tcp://localhost:7778"
    openclaw_workspace: str = str(Path.home() / ".openclaw" / "workspace")
    servo_channels: dict = field(default_factory=lambda: {
        "head_pan": 0, "head_tilt": 1,
        "left_wheel": 2, "right_wheel": 3,
        "left_arm": 4, "right_arm": 5,
    })
    max_motor_speed: int = 80


# ============================================================
# SIDECAR LISTENER - Consumes V-JEPA2 + SAM2 from kernel PUB
# ============================================================
class SidecarListener:
    """
    Subscribes to kernel PUB socket and collects the latest
    V-JEPA2 and SAM2 outputs. The brain reads these when building
    the VLM prompt so qwen3.5 gets richer context.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.vjepa2_latest = None   # Latest motion prediction
        self.sam2_latest = None     # Latest segmentation
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._listen, daemon=True, name="sidecars").start()

    def _listen(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.cfg.kernel_pub)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(timeout=200))
                if sub in socks:
                    data = sub.recv_json()
                    msg_type = data.get("type", "")
                    with self._lock:
                        if msg_type == "vjepa2_prediction":
                            self.vjepa2_latest = data
                        elif msg_type == "sam2_segmentation":
                            self.sam2_latest = data
            except Exception:
                time.sleep(0.1)

        sub.close()
        ctx.term()

    def get_context(self):
        """Return sidecar data for the VLM prompt. Thread-safe."""
        with self._lock:
            return {
                "vjepa2": self.vjepa2_latest,
                "sam2": self.sam2_latest,
            }

    def stop(self):
        self.running = False


# ============================================================
# SENSORY CORTEX
# ============================================================
class SensoryCortex:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.frame_left = None
        self.frame_right = None
        self.frame_ts = 0
        self.jpeg_left = b""
        self.jpeg_right = b""
        self.jpeg_combined = b""
        self._lock = threading.Lock()
        self.memory = deque(maxlen=10)
        self._caps = []

    def start(self):
        self.running = True
        for idx, cid in enumerate([self.cfg.camera_left, self.cfg.camera_right]):
            cap = self._open(cid)
            label = ["left", "right"][idx]
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
                cap.set(cv2.CAP_PROP_FPS, self.cfg.capture_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._caps.append(cap)
                print(f"  cam{cid} ({label}) OK")
            else:
                self._caps.append(None)
                print(f"  cam{cid} ({label}) FAILED")
        threading.Thread(target=self._loop, daemon=True, name="sensory").start()

    def _open(self, cid):
        cap = cv2.VideoCapture(cid, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        gst = (f"v4l2src device=/dev/video{cid} ! "
               f"video/x-raw,width={self.cfg.frame_width},height={self.cfg.frame_height},"
               f"framerate={self.cfg.capture_fps}/1 ! videoconvert ! "
               f"video/x-raw,format=BGR ! appsink drop=1")
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return cap if cap.isOpened() else None

    def _loop(self):
        while self.running:
            for i, cap in enumerate(self._caps):
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        if i == 0: self.frame_left = frame
                        else: self.frame_right = frame
            if self.frame_left is not None or self.frame_right is not None:
                self.frame_ts = time.time()
                self._encode()
                if self.frame_left is not None:
                    self.memory.append(self.frame_left.copy())
            time.sleep(1.0 / self.cfg.capture_fps)

    def _encode(self):
        with self._lock:
            p = [cv2.IMWRITE_JPEG_QUALITY, 75]
            if self.frame_left is not None:
                _, b = cv2.imencode('.jpg', self.frame_left, p)
                self.jpeg_left = b.tobytes()
            if self.frame_right is not None:
                _, b = cv2.imencode('.jpg', self.frame_right, p)
                self.jpeg_right = b.tobytes()
            if self.frame_left is not None and self.frame_right is not None:
                _, b = cv2.imencode('.jpg', np.hstack([self.frame_left, self.frame_right]), p)
                self.jpeg_combined = b.tobytes()
            elif self.frame_left is not None:
                self.jpeg_combined = self.jpeg_left

    def get_jpeg(self, cam="combined"):
        with self._lock:
            return {"left": self.jpeg_left, "right": self.jpeg_right}.get(cam, self.jpeg_combined)

    def get_vlm_snapshot(self):
        snap = {"timestamp": self.frame_ts, "images": []}
        for name, frame in [("left", self.frame_left), ("right", self.frame_right)]:
            if frame is not None:
                small = cv2.resize(frame, (self.cfg.vlm_width, self.cfg.vlm_height))
                _, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                snap["images"].append({"eye": name, "base64": base64.b64encode(jpg.tobytes()).decode()})
        snap["motion_detected"] = self._motion()
        return snap

    def _motion(self):
        if len(self.memory) < 2: return False
        d = cv2.absdiff(cv2.cvtColor(self.memory[-2], cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(self.memory[-1], cv2.COLOR_BGR2GRAY))
        return float(np.mean(d)) > 15.0

    def stop(self):
        self.running = False
        for c in self._caps:
            if c: c.release()


# ============================================================
# FRONTAL LOBE - VLM with sidecar context
# ============================================================
class FrontalLobe:
    SYS = """You are TEELA, a humanoid robot brain. You receive camera images plus data from two perception subsystems:
- V-JEPA2: predicts what ACTION is happening in the scene (e.g. "pushing something", "picking up")
- SAM2: segments objects and gives their positions (bounding boxes, areas, centers)

Your body: stereo cameras, head pan/tilt, differential drive wheels, two arms, speaker.

RESPOND ONLY with JSON:
{"scene_description":"what you see","objects_detected":["objects"],"threat_level":"none|low|medium|high","emotional_state":"curious|alert|calm|cautious|excited|happy","actions":[{"type":"move|look|speak|gesture|stop","target":"what","params":{"direction":"forward|backward|left|right|up|down","speed":0-100,"duration_ms":500}}],"reasoning":"why"}

RULES:
- Use V-JEPA2 action predictions to understand what's happening around you
- Use SAM2 object data to know WHERE things are (left/right/center, near/far by area)
- Person detected -> look, wave, greet by speaking
- Path blocked (large object centered) -> stop + look for alternatives
- Nothing interesting -> explore slowly, look around
- Max speed 80. Cautious movement. Danger -> STOP + warn
- Motion/action detected -> investigate
- Narrate via speak. Chain actions OK. JSON ONLY."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_decision = None
        self.decision_count = 0
        self.recent = deque(maxlen=3)

    def analyze_and_decide(self, snap, sidecar_ctx):
        if not snap["images"]:
            return self._default("no_cameras")

        # Build enriched prompt with sidecar data
        parts = []
        if snap.get("motion_detected"):
            parts.append("MOTION DETECTED in camera feed!")

        vjepa = sidecar_ctx.get("vjepa2")
        if vjepa and time.time() - vjepa.get("timestamp", 0) < 10:
            parts.append(f"V-JEPA2 predicts action: '{vjepa['action']}' (confidence: {vjepa['confidence']})")

        sam = sidecar_ctx.get("sam2")
        if sam and time.time() - sam.get("timestamp", 0) < 10:
            n = sam["num_objects"]
            parts.append(f"SAM2 detected {n} objects:")
            for obj in sam.get("objects", [])[:5]:
                cx, cy = obj.get("center", [0, 0])
                area = obj.get("area", 0)
                side = "left" if cx < 320 else "right" if cx > 320 else "center"
                size = "large" if area > 50000 else "medium" if area > 10000 else "small"
                parts.append(f"  - {size} object at {side} (area={area})")

        if self.recent:
            parts.append(f"Your previous action: {self.recent[-1].get('reasoning', '')}")

        parts.append(f"{len(snap['images'])} camera(s) active. Analyze and decide.")
        prompt = "\n".join(parts)
        imgs = [i["base64"] for i in snap["images"]]

        try:
            r = requests.post(
                f"{self.cfg.ollama_url}/api/chat",
                json={
                    "model": self.cfg.vlm_model,
                    "messages": [
                        {"role": "system", "content": self.SYS},
                        {"role": "user", "content": prompt, "images": imgs}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512}
                },
                timeout=self.cfg.vlm_timeout)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            d = self._parse(content)
            if d:
                self.last_decision = d
                self.decision_count += 1
                self.recent.append(d)
                return d
            print(f"  warn: VLM unparseable: {content[:100]}")
            return self._default("parse_error")
        except requests.Timeout: return self._default("timeout")
        except requests.ConnectionError: return self._default("no_ollama")
        except Exception as e:
            print(f"  VLM error: {e}")
            return self._default("error")

    def _parse(self, c):
        c = c.strip()
        if c.startswith("```"):
            lines = c.split("\n")
            c = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            d = json.loads(c)
            if "actions" in d and isinstance(d["actions"], list): return d
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', c)
            if m:
                try: return json.loads(m.group())
                except: pass
        return None

    def _default(self, reason):
        return {"scene_description": f"Unable to analyze ({reason})", "objects_detected": [],
                "threat_level": "low", "emotional_state": "cautious",
                "actions": [{"type": "stop", "target": "all", "params": {}}],
                "reasoning": f"Safety stop: {reason}"}


# ============================================================
# CONVERSATION ENGINE - Non-blocking continuous chat
# ============================================================
class ConversationEngine:
    """
    Separate thread for user conversation. Never blocks the perception loop.
    Users can talk via the web dashboard or OpenClaw channels.
    The VLM gets the current scene context so it can answer questions
    about what it sees.
    """
    def __init__(self, cfg, brain):
        self.cfg = cfg
        self.brain = brain
        self.running = False
        self.chat_history = deque(maxlen=20)  # Rolling conversation window
        self.pending_messages = deque()        # Incoming user messages
        self.latest_response = None
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._respond_loop, daemon=True, name="conversation").start()

    def send_message(self, text):
        """Queue a user message (called from web API or OpenClaw)."""
        with self._lock:
            self.pending_messages.append({"role": "user", "content": text, "time": time.time()})

    def get_latest_response(self):
        with self._lock:
            return self.latest_response

    def _respond_loop(self):
        while self.running:
            msg = None
            with self._lock:
                if self.pending_messages:
                    msg = self.pending_messages.popleft()

            if msg:
                response = self._generate_response(msg["content"])
                with self._lock:
                    self.chat_history.append(msg)
                    self.chat_history.append({"role": "assistant", "content": response, "time": time.time()})
                    self.latest_response = response

                # Speak the response
                self.brain.motor.execute_decision({
                    "actions": [{"type": "speak", "target": response[:200], "params": {}}]
                })
            else:
                time.sleep(0.2)

    def _generate_response(self, user_text):
        """Generate a conversational response with scene awareness."""
        # Build context about what TEELA currently sees
        scene_ctx = ""
        if self.brain.frontal.last_decision:
            d = self.brain.frontal.last_decision
            scene_ctx = (f"You currently see: {d.get('scene_description', 'unknown')}. "
                        f"Objects: {', '.join(d.get('objects_detected', []))}. "
                        f"You feel {d.get('emotional_state', 'calm')}.")

        sidecar = self.brain.sidecars.get_context()
        if sidecar.get("vjepa2"):
            scene_ctx += f" V-JEPA2 detects action: {sidecar['vjepa2'].get('action', '?')}."
        if sidecar.get("sam2"):
            scene_ctx += f" SAM2 sees {sidecar['sam2'].get('num_objects', 0)} objects."

        system = (f"You are TEELA, a friendly humanoid robot. You are having a live conversation "
                  f"with a human near you. Keep responses concise (1-3 sentences) and natural. "
                  f"You can see through your cameras. {scene_ctx}")

        messages = [{"role": "system", "content": system}]
        # Add recent chat history
        for h in list(self.chat_history)[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_text})

        try:
            r = requests.post(
                f"{self.cfg.ollama_url}/api/chat",
                json={"model": self.cfg.vlm_model,
                      "messages": messages,
                      "stream": False,
                      "options": {"temperature": 0.7, "num_predict": 256}},
                timeout=30)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "I'm having trouble thinking right now.")
        except Exception as e:
            return f"Sorry, I can't respond right now. ({e})"

    def stop(self):
        self.running = False


# ============================================================
# MOTOR CORTEX
# ============================================================
class MotorCortex:
    DRIVE = {"forward": (1, 1), "backward": (-1, -1), "left": (-1, 1), "right": (1, -1)}
    LOOK = {"left": ("head_pan", -30), "right": ("head_pan", 30),
            "up": ("head_tilt", -20), "down": ("head_tilt", 20)}

    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = zmq.Context()
        self.sock = None
        self.ok = False

    def connect(self):
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt_string(zmq.IDENTITY, "motor_cortex")
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.cfg.kernel_router)
        self.ok = True

    def execute_decision(self, decision):
        if not self.ok: return
        for act in decision.get("actions", []):
            at = act.get("type", "stop")
            p = act.get("params", {})
            d = p.get("direction", "forward")
            sp = min(p.get("speed", 30), self.cfg.max_motor_speed)
            dur = p.get("duration_ms", 500)
            if at == "move" and d in self.DRIVE:
                lm, rm = self.DRIVE[d]
                for motor, mult in [("left_wheel", lm), ("right_wheel", rm)]:
                    ch = self.cfg.servo_channels.get(motor)
                    if ch is not None:
                        self._tx({"type":"command","service_id":"motor_cortex","target":"motor","channel":ch,"speed":sp*mult,"duration_ms":dur})
            elif at == "look" and d in self.LOOK:
                servo, angle = self.LOOK[d]
                ch = self.cfg.servo_channels.get(servo)
                if ch is not None:
                    self._tx({"type":"command","service_id":"motor_cortex","target":"servo","channel":ch,"angle":angle})
            elif at == "stop":
                for m in ["left_wheel","right_wheel"]:
                    ch = self.cfg.servo_channels.get(m)
                    if ch is not None:
                        self._tx({"type":"command","service_id":"motor_cortex","target":"motor","channel":ch,"speed":0,"duration_ms":0})
            elif at == "gesture":
                ch = self.cfg.servo_channels.get("right_arm")
                if ch is not None:
                    for a in [45,-10,45,-10,0]:
                        self._tx({"type":"command","service_id":"motor_cortex","target":"servo","channel":ch,"angle":a})
                        time.sleep(0.3)
            elif at == "speak":
                self._tx({"type":"command","service_id":"motor_cortex","target":"speech","action":"speak","text":act.get("target","Hello")})

    def _tx(self, data):
        try: self.sock.send_json(data)
        except zmq.ZMQError: pass

    def stop(self):
        self.execute_decision({"actions":[{"type":"stop","target":"all","params":{}}]})
        if self.sock: self.sock.close()
        self.ctx.term()


# ============================================================
# WEB DASHBOARD (abbreviated - same HTML as before + chat box)
# ============================================================
DASH_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TEELA CortexOS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0f;color:#e0e0e0;height:100vh;overflow:hidden}
.hd{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:12px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #2a2a4a}
.hd h1{font-size:18px;background:linear-gradient(90deg,#00d4ff,#7b68ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hd .s{display:flex;align-items:center;gap:8px;font-size:12px;color:#8a8aaa}
.dt{width:8px;height:8px;border-radius:50%;background:#00ff88;animation:pu 2s infinite}
@keyframes pu{0%,100%{opacity:1}50%{opacity:.4}}
.mn{display:grid;grid-template-columns:1fr 340px;gap:12px;padding:12px;height:calc(100vh-48px)}
.cm{display:flex;flex-direction:column;gap:10px;min-height:0}
.cv{position:relative;background:#111;border-radius:10px;overflow:hidden;border:1px solid #2a2a4a;flex:1;min-height:0}
.cv img{width:100%;height:100%;object-fit:contain}
.lb{position:absolute;top:6px;left:10px;background:rgba(0,0,0,.8);padding:3px 8px;border-radius:5px;font-size:10px;color:#00d4ff;font-weight:700;z-index:1}
.rw{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-height:160px}
.sm{position:relative;background:#111;border-radius:8px;overflow:hidden;border:1px solid #2a2a4a}.sm img{width:100%;display:block}
.sb{display:flex;flex-direction:column;gap:10px;overflow-y:auto}
.pn{background:#12121e;border-radius:10px;border:1px solid #2a2a4a;padding:14px}
.pn h3{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#7b68ee;margin-bottom:10px;font-weight:700}
.ct{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.ct button{padding:11px;border:1px solid #3a3a5a;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:18px;cursor:pointer;transition:.15s}
.ct button:hover{background:#2a2a4e}.ct button:active{background:#7b68ee;transform:scale(.95)}
.ct .sp{background:#3a1020;border-color:#ff4444;grid-column:2}.ct .sp:hover{background:#ff4444}
.lg{font-family:monospace;font-size:11px;line-height:1.5;color:#888;max-height:120px;overflow-y:auto}
.lg .a{color:#00d4ff}.lg .b{color:#ff6b9d}.lg .c{color:#00ff88}
.ss{display:flex;justify-content:space-between;padding:3px 0;font-size:12px}.ss .l{color:#8a8aaa}.ss .v{color:#00d4ff;font-family:monospace}
.chat{display:flex;flex-direction:column;gap:6px;max-height:140px}
.chat-log{font-size:12px;overflow-y:auto;max-height:90px;line-height:1.5}
.chat-log .u{color:#7b68ee}.chat-log .t{color:#00ff88}
.vi{display:flex;gap:6px}.vi input{flex:1;padding:7px 10px;background:#1a1a2e;border:1px solid #3a3a5a;border-radius:7px;color:#e0e0e0;font-size:12px;outline:none}
.vi button{padding:7px 14px;background:#7b68ee;border:none;border-radius:7px;color:#fff;cursor:pointer;font-size:12px;font-weight:600}
@media(max-width:900px){.mn{grid-template-columns:1fr}}
</style></head><body>
<div class="hd"><h1>TEELA CortexOS</h1><div class="s"><div class="dt" id="dot"></div><span id="st">...</span><span style="color:#444">|</span><span style="color:#7b68ee" id="md">qwen3.5:397b-cloud</span></div></div>
<div class="mn"><div class="cm">
<div class="cv"><div class="lb">STEREO VIEW</div><img src="/stream/both"></div>
<div class="rw"><div class="sm"><div class="lb">LEFT EYE</div><img src="/stream/left"></div><div class="sm"><div class="lb">RIGHT EYE</div><img src="/stream/right"></div></div>
</div><div class="sb">
<div class="pn"><h3>Controls</h3><div class="ct"><div></div><button onclick="c('forward')">&#x2B06;&#xFE0F;</button><div></div><button onclick="c('left')">&#x2B05;&#xFE0F;</button><button class="sp" onclick="c('stop')">&#x23F9;</button><button onclick="c('right')">&#x27A1;&#xFE0F;</button><div></div><button onclick="c('backward')">&#x2B07;&#xFE0F;</button><div></div></div></div>
<div class="pn"><h3>Brain</h3>
<div class="ss"><span class="l">Cycle</span><span class="v" id="cy">0</span></div>
<div class="ss"><span class="l">Think</span><span class="v" id="tt">-</span></div>
<div class="ss"><span class="l">Emotion</span><span class="v" id="em">-</span></div>
<div class="ss"><span class="l">V-JEPA2</span><span class="v" id="vj">-</span></div>
<div class="ss"><span class="l">SAM2 Objs</span><span class="v" id="s2">-</span></div>
</div>
<div class="pn"><h3>Activity</h3><div class="lg" id="lg"><div style="color:#444">Waiting...</div></div></div>
<div class="pn"><h3>Talk to TEELA</h3>
<div class="chat"><div class="chat-log" id="cl"></div>
<div class="vi"><input id="ci" placeholder="Say something to TEELA..." onkeydown="if(event.key==='Enter')chat()"><button onclick="chat()">Send</button></div>
</div></div>
</div></div>
<script>
function c(v){fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:v})})}
function chat(){var i=document.getElementById('ci'),cl=document.getElementById('cl');if(!i.value.trim())return;
var t=i.value.trim();i.value='';
var e=document.createElement('div');e.innerHTML='<span class="u">You:</span> '+t;cl.appendChild(e);cl.scrollTop=1e9;
fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})}).then(function(r){return r.json()}).then(function(d){
var r=document.createElement('div');r.innerHTML='<span class="t">TEELA:</span> '+(d.response||'...');cl.appendChild(r);cl.scrollTop=1e9})}
var lg=document.getElementById('lg'),lc=0;
function u(){fetch('/api/status').then(function(r){return r.json()}).then(function(s){
document.getElementById('dot').style.background=s.running?'#00ff88':'#ff4444';
document.getElementById('st').textContent=s.running?'LIVE':'OFF';
document.getElementById('cy').textContent=s.cycle;
document.getElementById('md').textContent=s.model;
document.getElementById('tt').textContent=s.avg_think_time?s.avg_think_time.toFixed(1)+'s':'-';
document.getElementById('vj').textContent=s.vjepa2_action||'-';
document.getElementById('s2').textContent=s.sam2_objects!=null?s.sam2_objects:'-';
if(s.last_decision&&s.cycle>lc){lc=s.cycle;var d=s.last_decision;
document.getElementById('em').textContent=d.emotional_state||'-';
var a=(d.actions||[]).map(function(x){return x.type}).join(',');
var e=document.createElement('div');
e.innerHTML='<span class="a">['+s.cycle+']</span> <span class="b">'+(d.emotional_state||'?')+'</span> <span class="c">'+a+'</span> '+(d.scene_description||'').substring(0,60);
lg.appendChild(e);lg.scrollTop=1e9;while(lg.children.length>40)lg.removeChild(lg.firstChild)}
}).catch(function(){document.getElementById('dot').style.background='#ff4444';document.getElementById('st').textContent='OFF'})}
setInterval(u,2000);u();
document.addEventListener('keydown',function(e){var m={ArrowUp:'forward',ArrowDown:'backward',ArrowLeft:'left',ArrowRight:'right',' ':'stop',Escape:'stop'};if(m[e.key]){e.preventDefault();c(m[e.key])}});
</script></body></html>"""


class WebDashboard:
    def __init__(self, cfg, brain):
        self.cfg = cfg
        self.brain = brain
        self.server = None

    def start(self):
        dash = self
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200);self.send_header('Content-Type','text/html');self.end_headers()
                    self.wfile.write(DASH_HTML.encode())
                elif self.path.startswith('/stream/'):
                    cam = self.path.split('/')[-1]
                    self.send_response(200)
                    self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control','no-cache');self.end_headers()
                    while dash.brain.running:
                        f = dash.brain.sensory.get_jpeg(cam)
                        if f:
                            try:
                                self.wfile.write(f'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(f)}\r\n\r\n'.encode()+f+b'\r\n')
                            except: break
                        time.sleep(1/15)
                elif self.path == '/api/status':
                    sc = dash.brain.sidecars.get_context()
                    s = {"cycle":dash.brain.cycle_count,"running":dash.brain.running,
                         "last_decision":dash.brain.frontal.last_decision,"model":dash.brain.cfg.vlm_model,
                         "avg_think_time":dash.brain.total_think_time/max(1,dash.brain.cycle_count),
                         "vjepa2_action":sc["vjepa2"]["action"] if sc.get("vjepa2") else None,
                         "sam2_objects":sc["sam2"]["num_objects"] if sc.get("sam2") else None}
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Access-Control-Allow-Origin','*');self.end_headers()
                    self.wfile.write(json.dumps(s,default=str).encode())
                else: self.send_error(404)
            def do_POST(self):
                ln=int(self.headers.get('Content-Length',0))
                body=json.loads(self.rfile.read(ln)) if ln else {}
                if self.path=='/api/command':
                    cmd=body.get("command","")
                    am={"stop":[{"type":"stop","target":"all","params":{}}],
                        "forward":[{"type":"move","target":"body","params":{"direction":"forward","speed":40,"duration_ms":1000}}],
                        "backward":[{"type":"move","target":"body","params":{"direction":"backward","speed":40,"duration_ms":1000}}],
                        "left":[{"type":"move","target":"body","params":{"direction":"left","speed":30,"duration_ms":500}}],
                        "right":[{"type":"move","target":"body","params":{"direction":"right","speed":30,"duration_ms":500}}]}
                    if cmd in am: dash.brain.motor.execute_decision({"actions":am[cmd]})
                    elif cmd=="say": dash.brain.motor.execute_decision({"actions":[{"type":"speak","target":body.get("text","Hello"),"params":{}}]})
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Access-Control-Allow-Origin','*');self.end_headers();self.wfile.write(b'{"ok":true}')
                elif self.path=='/api/chat':
                    text=body.get("text","")
                    if text: dash.brain.conversation.send_message(text)
                    # Wait briefly for response
                    time.sleep(0.1)
                    for _ in range(50):  # Wait up to 5s
                        resp = dash.brain.conversation.get_latest_response()
                        if resp: break
                        time.sleep(0.1)
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Access-Control-Allow-Origin','*');self.end_headers()
                    self.wfile.write(json.dumps({"response":resp or "Thinking..."}).encode())
                else: self.send_error(404)
            def do_OPTIONS(self):
                self.send_response(204);self.send_header('Access-Control-Allow-Origin','*')
                self.send_header('Access-Control-Allow-Methods','GET,POST');self.send_header('Access-Control-Allow-Headers','Content-Type');self.end_headers()
            def log_message(self,*a): pass

        self.server=HTTPServer((self.cfg.web_host,self.cfg.web_port),H)
        self.server.socket.settimeout(1)
        def serve():
            print(f"  web: http://0.0.0.0:{self.cfg.web_port}")
            while dash.brain.running:
                try: self.server.handle_request()
                except: pass
        threading.Thread(target=serve,daemon=True,name="web").start()

    def stop(self):
        if self.server: self.server.server_close()


# ============================================================
# MAIN BRAIN
# ============================================================
class TEELABrain:
    def __init__(self, cfg=None):
        self.cfg = cfg or RobotConfig()
        self.sensory = SensoryCortex(self.cfg)
        self.frontal = FrontalLobe(self.cfg)
        self.motor = MotorCortex(self.cfg)
        self.sidecars = SidecarListener(self.cfg)
        self.conversation = ConversationEngine(self.cfg, self)
        self.dashboard = WebDashboard(self.cfg, self)
        self.running = False
        self.cycle_count = 0
        self.total_think_time = 0

    def start(self):
        print("=" * 56)
        print("  TEELA CortexOS (V-JEPA2 + SAM2 + Conversation)")
        print(f"  VLM:   {self.cfg.vlm_model}")
        print(f"  Cams:  /dev/video{self.cfg.camera_left}, /dev/video{self.cfg.camera_right}")
        print(f"  Web:   http://0.0.0.0:{self.cfg.web_port}")
        print("=" * 56)

        if not self._check_ollama():
            print(f"\n  Cannot reach Ollama. Run: ollama serve && ollama pull {self.cfg.vlm_model}")
            return

        self.sensory.start()
        self.motor.connect()
        self.sidecars.start()
        self.conversation.start()
        self.running = True
        self.dashboard.start()
        self._start_hb()

        print("\n  Waiting for cameras + sidecars...")
        time.sleep(3)
        print("  TEELA IS LIVE\n")

        try:
            self._loop()
        except KeyboardInterrupt:
            print("\n  Shutdown")
        finally:
            self.stop()

    def _check_ollama(self):
        try:
            requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=5)
            t = requests.post(f"{self.cfg.ollama_url}/api/chat",
                json={"model":self.cfg.vlm_model,"messages":[{"role":"user","content":"OK"}],
                      "stream":False,"options":{"num_predict":5}}, timeout=20)
            if t.status_code == 200:
                print(f"  Ollama: {self.cfg.vlm_model} OK")
                return True
            return False
        except: return False

    def _loop(self):
        while self.running:
            t0 = time.time()
            self.cycle_count += 1
            snap = self.sensory.get_vlm_snapshot()
            if not snap["images"]:
                time.sleep(self.cfg.analysis_interval); continue
            sidecar_ctx = self.sidecars.get_context()
            ts = time.time()
            dec = self.frontal.analyze_and_decide(snap, sidecar_ctx)
            tt = time.time() - ts
            self.total_think_time += tt
            if dec: self.motor.execute_decision(dec)

            # Log
            avg = self.total_think_time / self.cycle_count
            acts = ",".join(a.get("type","?") for a in dec.get("actions",[])) if dec else "?"
            vj = sidecar_ctx.get("vjepa2",{}).get("action","?") if sidecar_ctx.get("vjepa2") else "-"
            s2 = sidecar_ctx.get("sam2",{}).get("num_objects","?") if sidecar_ctx.get("sam2") else "-"
            sc = (dec.get("scene_description","") if dec else "")[:45]
            print(f"[{self.cycle_count:>4}] think:{tt:.1f}s vj:{vj} s2:{s2}obj {acts} | {sc}")

            sl = max(0, self.cfg.analysis_interval - (time.time() - t0))
            if sl > 0: time.sleep(sl)

    def _start_hb(self):
        ctx = zmq.Context()
        hb = ctx.socket(zmq.DEALER)
        hb.setsockopt_string(zmq.IDENTITY, "cortex_live")
        hb.setsockopt(zmq.LINGER, 0)
        hb.connect(self.cfg.kernel_router)
        def loop():
            while self.running:
                try: hb.send_json({"type":"heartbeat","service_id":"cortex","cycle":self.cycle_count,"timestamp":time.time()})
                except: pass
                time.sleep(3)
        threading.Thread(target=loop,daemon=True,name="hb").start()

    def stop(self):
        self.running = False
        self.conversation.stop()
        self.sidecars.stop()
        self.dashboard.stop()
        self.motor.stop()
        self.sensory.stop()
        print("  TEELA offline")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:397b-cloud")
    p.add_argument("--cam-left", type=int, default=0)
    p.add_argument("--cam-right", type=int, default=2)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--resolution", default="640x480")
    p.add_argument("--vlm-resolution", default="320x240")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    a = p.parse_args()
    w,h = map(int, a.resolution.split("x"))
    vw,vh = map(int, a.vlm_resolution.split("x"))
    TEELABrain(RobotConfig(vlm_model=a.model,camera_left=a.cam_left,camera_right=a.cam_right,
        analysis_interval=a.interval,frame_width=w,frame_height=h,vlm_width=vw,vlm_height=vh,
        web_port=a.port,ollama_url=a.ollama_url)).start()
