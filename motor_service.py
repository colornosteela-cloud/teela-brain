#!/usr/bin/env python3
"""
TEELA Motor Service
====================
Receives motor/servo/speech commands from the kernel PUB socket.
Translates them into hardware PWM signals via:
  - Adafruit PCA9685 I2C servo driver (most common for humanoids)
  - Jetson GPIO for DC motors via H-bridge (L298N, etc.)
  - espeak for text-to-speech

Falls back to simulation mode if hardware libs aren't installed,
so you can test the full pipeline without physical servos.

Sends heartbeats to the kernel for auto-restart monitoring.
"""

import zmq
import json
import time
import threading
import subprocess

# ---------- CONFIG ----------
KERNEL_ROUTER = "tcp://localhost:7777"
KERNEL_PUB = "tcp://localhost:7778"
SERVICE_ID = "motor"
HEARTBEAT_INTERVAL = 3

# PCA9685 servo driver
PCA9685_CHANNELS = 16
SERVO_MIN_PULSE = 500    # microseconds
SERVO_MAX_PULSE = 2500

# DC motor pin mapping (Jetson GPIO BOARD numbering)
# channel_id -> {enable_pin, input1_pin, input2_pin}
MOTOR_PINS = {
    2: {"en": 32, "in1": 29, "in2": 31},   # left_wheel
    3: {"en": 33, "in1": 35, "in2": 37},   # right_wheel
}

# ---------- HARDWARE IMPORTS (graceful) ----------
try:
    from adafruit_servokit import ServoKit
    HAS_SERVOKIT = True
except ImportError:
    HAS_SERVOKIT = False

try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False


class HardwareDriver:
    """
    Wraps all hardware access. Auto-detects what's available
    and falls back to simulation for anything missing.
    """

    def __init__(self):
        self.servo_kit = None
        self.sim_mode = False
        self._motor_pwms = {}

        # --- Servo driver (PCA9685 via I2C) ---
        if HAS_SERVOKIT:
            try:
                self.servo_kit = ServoKit(channels=PCA9685_CHANNELS)
                for i in range(PCA9685_CHANNELS):
                    self.servo_kit.servo[i].set_pulse_width_range(
                        SERVO_MIN_PULSE, SERVO_MAX_PULSE)
                print("[motor] PCA9685 servo driver OK")
            except Exception as e:
                print(f"[motor] PCA9685 failed: {e} (simulation mode)")
                self.sim_mode = True
        else:
            print("[motor] No adafruit_servokit (simulation mode)")
            self.sim_mode = True

        # --- GPIO for DC motors ---
        if HAS_GPIO and not self.sim_mode:
            try:
                GPIO.setmode(GPIO.BOARD)
                GPIO.setwarnings(False)
                for ch, pins in MOTOR_PINS.items():
                    GPIO.setup(pins["en"], GPIO.OUT)
                    GPIO.setup(pins["in1"], GPIO.OUT)
                    GPIO.setup(pins["in2"], GPIO.OUT)
                    pwm = GPIO.PWM(pins["en"], 1000)  # 1kHz PWM
                    pwm.start(0)
                    self._motor_pwms[ch] = pwm
                print("[motor] GPIO motor driver OK")
            except Exception as e:
                print(f"[motor] GPIO failed: {e}")

    def set_servo(self, channel, angle):
        """
        Set servo to angle.
        Input angle is -90 to +90 (centered at 0).
        PCA9685 expects 0-180.
        """
        hw_angle = max(0, min(180, angle + 90))

        if self.sim_mode:
            print(f"  [SIM] servo ch{channel} -> {hw_angle}deg ({angle:+d} raw)")
            return

        if self.servo_kit and 0 <= channel < PCA9685_CHANNELS:
            try:
                self.servo_kit.servo[channel].angle = hw_angle
            except Exception as e:
                print(f"  [motor] servo ch{channel} error: {e}")

    def set_motor(self, channel, speed, duration_ms=500):
        """
        Set DC motor speed.
        speed: -100 to 100 (negative = reverse, 0 = stop)
        duration_ms: auto-stop after this time (0 = manual stop only)
        """
        if self.sim_mode:
            print(f"  [SIM] motor ch{channel} speed={speed} dur={duration_ms}ms")
            if duration_ms > 0 and speed != 0:
                threading.Timer(duration_ms / 1000.0,
                    lambda: print(f"  [SIM] motor ch{channel} auto-stop")).start()
            return

        pins = MOTOR_PINS.get(channel)
        if not pins or not HAS_GPIO:
            return

        # Set direction
        if speed > 0:
            GPIO.output(pins["in1"], GPIO.HIGH)
            GPIO.output(pins["in2"], GPIO.LOW)
        elif speed < 0:
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.HIGH)
        else:
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.LOW)

        # Set speed (PWM duty cycle 0-100)
        pwm = self._motor_pwms.get(channel)
        if pwm:
            pwm.ChangeDutyCycle(min(abs(speed), 100))

        # Auto-stop timer
        if duration_ms > 0 and speed != 0:
            threading.Timer(
                duration_ms / 1000.0,
                lambda: self.set_motor(channel, 0, 0)
            ).start()

    def emergency_stop(self):
        """Kill all motors immediately."""
        print("[motor] EMERGENCY STOP")
        for ch in MOTOR_PINS:
            self.set_motor(ch, 0, 0)
        # Also center all servos
        if self.servo_kit:
            for i in range(PCA9685_CHANNELS):
                try:
                    self.servo_kit.servo[i].angle = 90
                except Exception:
                    pass

    def speak(self, text):
        """Text-to-speech via espeak (pre-installed on most Jetson images)."""
        if not text:
            return
        try:
            # Non-blocking: fire and forget
            subprocess.Popen(
                ["espeak", "-v", "en+f3", "-s", "150", "--", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            # espeak not installed, try pico2wave
            try:
                subprocess.Popen(
                    ["pico2wave", "-w", "/tmp/teela_tts.wav", "--", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                subprocess.Popen(
                    ["aplay", "/tmp/teela_tts.wav"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except FileNotFoundError:
                print(f"  [TTS unavailable] Would say: {text[:80]}")

    def cleanup(self):
        self.emergency_stop()
        if HAS_GPIO and not self.sim_mode:
            try:
                for pwm in self._motor_pwms.values():
                    pwm.stop()
                GPIO.cleanup()
            except Exception:
                pass


class MotorService:
    """
    Kernel-managed service:
      1. Subscribes to commands from kernel PUB socket
      2. Executes motor/servo/speech commands on hardware
      3. Sends heartbeats to kernel ROUTER
    """

    def __init__(self):
        self.ctx = zmq.Context()
        self.running = False
        self.driver = HardwareDriver()
        self.cmd_count = 0

    def start(self):
        print(f"[motor] Motor service starting")
        self.running = True

        # Subscribe to commands from kernel PUB
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(KERNEL_PUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Heartbeat via DEALER to kernel ROUTER
        self.hb = self.ctx.socket(zmq.DEALER)
        self.hb.setsockopt_string(zmq.IDENTITY, SERVICE_ID)
        self.hb.setsockopt(zmq.LINGER, 0)
        self.hb.connect(KERNEL_ROUTER)

        threading.Thread(target=self._heartbeat, daemon=True).start()

        print(f"[motor] Ready (sim={self.driver.sim_mode})")
        self._command_loop()

    def _command_loop(self):
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(timeout=100))
                if self.sub in socks:
                    data = self.sub.recv_json()

                    # Only handle command messages
                    if data.get("type") != "command":
                        continue

                    target = data.get("target")
                    self.cmd_count += 1

                    if target == "servo":
                        self.driver.set_servo(
                            data.get("channel", 0),
                            data.get("angle", 0))

                    elif target == "motor":
                        self.driver.set_motor(
                            data.get("channel", 0),
                            data.get("speed", 0),
                            data.get("duration_ms", 500))

                    elif target == "speech":
                        self.driver.speak(data.get("text", ""))

                    elif target == "emergency_stop":
                        self.driver.emergency_stop()

            except json.JSONDecodeError:
                pass
            except zmq.ZMQError as e:
                if self.running:
                    print(f"[motor] ZMQ error: {e}")
                    time.sleep(0.1)
            except Exception as e:
                print(f"[motor] Error: {e}")
                time.sleep(0.5)

    def _heartbeat(self):
        while self.running:
            try:
                self.hb.send_json({
                    "type": "heartbeat",
                    "service_id": SERVICE_ID,
                    "commands": self.cmd_count,
                    "sim": self.driver.sim_mode,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            time.sleep(HEARTBEAT_INTERVAL)

    def stop(self):
        self.running = False
        self.driver.cleanup()
        self.sub.close()
        self.hb.close()
        self.ctx.term()
        print("[motor] Stopped")


if __name__ == "__main__":
    svc = MotorService()
    try:
        svc.start()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
