#!/usr/bin/env python3
"""
TEELA Kernel - Hypothalamus + Message Bus
==========================================
Manages ALL services: camera, motor, cortex, vjepa2, sam2
Auto-restarts crashed services. Routes messages between them.
Fixed: recv_multipart frame handling, thread health, restart cooldown.
"""

import zmq
import threading
import time
import json
import subprocess
import os
import signal
from pathlib import Path

SERVICES_DIR = Path(__file__).parent / "services"

SERVICE_CONFIG = {
    'camera':  {'script': 'camera_service_mock.py', 'critical': True,  'timeout': 12},
    'motor':   {'script': 'motor_service.py',       'critical': True,  'timeout': 12},
    'cortex':  {'script': 'cortex.py',              'critical': True,  'timeout': 12},
    'vjepa2':  {'script': 'vjepa2_service.py',      'critical': False, 'timeout': 30},
    'sam2':    {'script': 'sam2_service.py',         'critical': False, 'timeout': 30},
}

MAX_RESTARTS = 10
RESTART_COOLDOWN = 5.0
MONITOR_INTERVAL = 2


class Hypothalamus:
    def __init__(self):
        self.context = zmq.Context()
        self.running = False
        self.service_vitals = {}
        self.service_processes = {}
        self.restart_count = {n: 0 for n in SERVICE_CONFIG}
        self.last_restart_time = {}
        self.router = None
        self.pub = None
        self._router_thread = None
        self._monitor_thread = None

    def start(self):
        print("=" * 56)
        print("  TEELA Kernel (Hypothalamus)")
        print("  Services:", ", ".join(SERVICE_CONFIG.keys()))
        print("  Ports: 7777 (ROUTER), 7778 (PUB)")
        print("=" * 56)

        try:
            self.router = self.context.socket(zmq.ROUTER)
            self.router.setsockopt(zmq.LINGER, 0)
            self.router.setsockopt(zmq.ROUTER_MANDATORY, 0)
            self.router.bind("tcp://*:7777")

            self.pub = self.context.socket(zmq.PUB)
            self.pub.setsockopt(zmq.LINGER, 0)
            self.pub.bind("tcp://*:7778")

            print("  Message bus active")
            self.running = True

            self._start_all_services()

            self._router_thread = threading.Thread(target=self._message_router, daemon=True, name="router")
            self._monitor_thread = threading.Thread(target=self._monitor_vitals, daemon=True, name="monitor")
            self._router_thread.start()
            self._monitor_thread.start()

            print("  Monitoring...\n")
            while self.running:
                time.sleep(1)
                self._check_threads()

        except KeyboardInterrupt:
            print("\nShutdown requested")
        except Exception as e:
            print(f"Kernel error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def _check_threads(self):
        if self._router_thread and not self._router_thread.is_alive():
            print("  warn: router thread died, restarting")
            self._router_thread = threading.Thread(target=self._message_router, daemon=True, name="router")
            self._router_thread.start()
        if self._monitor_thread and not self._monitor_thread.is_alive():
            print("  warn: monitor thread died, restarting")
            self._monitor_thread = threading.Thread(target=self._monitor_vitals, daemon=True, name="monitor")
            self._monitor_thread.start()

    def _start_all_services(self):
        for sid, cfg in SERVICE_CONFIG.items():
            self._start_service(sid, cfg)
            time.sleep(0.5)

    def _start_service(self, sid, cfg):
        script = SERVICES_DIR / cfg['script']
        if not script.exists():
            print(f"  {sid}: script not found at {script}")
            return False
        try:
            print(f"  Starting {sid}...")
            proc = subprocess.Popen(
                ["python3", str(script)],
                cwd=str(Path(__file__).parent),
                preexec_fn=os.setsid)
            self.service_processes[sid] = proc
            self.service_vitals[sid] = time.time()
            print(f"  {sid} PID {proc.pid}")
            return True
        except Exception as e:
            print(f"  {sid}: start failed: {e}")
            return False

    def _message_router(self):
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)
        while self.running:
            try:
                socks = dict(poller.poll(timeout=100))
                if self.router in socks:
                    frames = self.router.recv_multipart()
                    if len(frames) < 2:
                        continue
                    msg = frames[-1]
                    try:
                        data = json.loads(msg)
                        msg_type = data.get('type')
                        sid = data.get('service_id')
                        if msg_type == 'heartbeat' and sid:
                            self.service_vitals[sid] = time.time()
                        self.pub.send_json(data)
                    except json.JSONDecodeError:
                        pass
            except zmq.ZMQError as e:
                if self.running:
                    print(f"  router zmq error: {e}")
                    time.sleep(0.1)
            except Exception as e:
                if self.running:
                    print(f"  router error: {e}")
                    time.sleep(0.5)

    def _monitor_vitals(self):
        time.sleep(8)  # Grace period for model loading (V-JEPA2/SAM2 take time)
        while self.running:
            now = time.time()
            for sid, cfg in SERVICE_CONFIG.items():
                proc = self.service_processes.get(sid)
                last_hb = self.service_vitals.get(sid, 0)
                age = now - last_hb
                dead = proc is not None and proc.poll() is not None
                frozen = age > cfg['timeout'] and last_hb > 0

                if dead or frozen:
                    reason = "DIED" if dead else "FROZEN"
                    if dead:
                        print(f"\n  {sid}: {reason} (exit={proc.returncode})")
                    else:
                        print(f"\n  {sid}: {reason} (no heartbeat {age:.0f}s)")
                    self._restart_service(sid, cfg)
            time.sleep(MONITOR_INTERVAL)

    def _restart_service(self, sid, cfg):
        last = self.last_restart_time.get(sid, 0)
        if time.time() - last < RESTART_COOLDOWN:
            return
        if self.restart_count[sid] >= MAX_RESTARTS:
            print(f"  {sid}: max restarts reached, giving up")
            return

        self.restart_count[sid] += 1
        self.last_restart_time[sid] = time.time()
        n = self.restart_count[sid]
        print(f"  Restarting {sid} (#{n})...")

        proc = self.service_processes.get(sid)
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=2)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if self._start_service(sid, cfg):
            print(f"  {sid}: RESTARTED")
        else:
            print(f"  {sid}: restart failed")

    def stop(self):
        self.running = False
        for sid, proc in self.service_processes.items():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1)
        for s in (self.router, self.pub):
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        try:
            self.context.term()
        except Exception:
            pass
        print("Kernel stopped")


if __name__ == "__main__":
    Hypothalamus().start()
