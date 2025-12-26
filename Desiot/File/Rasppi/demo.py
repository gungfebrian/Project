# ==========================================
# RASPBERRY PI SUPER SERVER (Leader's Version)
# ==========================================
# 1. Hosts Web Dashboard (Video + Buttons + Voice)
# 2. Runs Standard YOLOv8 (Better Accuracy)
# 3. Processes ALL Logic (Voice & Manual) locally
# 4. FIXED: Restored Tracking Logic & Smart Search


import os
import time
import signal
import sys


# Force kill any existing camera processes
os.system("pkill -9 -f libcamera")


from flask import Flask, render_template_string, request, jsonify, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import serial
import numpy as np
import threading
import speech_recognition as sr
from gtts import gTTS
import pygame
import json


# =======================
# CONFIGURATION
# =======================
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200


# --- OPTIMIZATION ---
HEADLESS_MODE = False     
SKIP_FRAMES = 3           # Run AI every 3 frames to save CPU
CONF_THRESHOLD = 0.4     
MODEL_FILE = 'yolov8n.pt' # Standard Model for Accuracy


# =======================
# HARDWARE SETUP
# =======================
print(f"🚀 Loading Standard YOLO Model ({MODEL_FILE})...")
try:
   model = YOLO(MODEL_FILE)
except Exception as e:
   print(f"❌ Model Error: {e}")
   model = None


try:
   ser = serial.Serial(SERIAL_PORT, baudrate=BAUDRATE, timeout=1)
   ser.flush()
   print(f"✅ Connected to ESP32 on {SERIAL_PORT}")
except:
   print("⚠️ WARNING: ESP32 NOT CONNECTED (Simulation Mode)")
   ser = None


try:
    picam2 = Picamera2()
    # [CRITICAL FIX] Anti-Blur Configuration
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={
            "FrameDurationLimits": (33333, 33333), # Mengunci di 30 FPS
            "AnalogueGain": 10.0,                   # Mencegah noise/bintik
            "ExposureTime": 20000             # <--- DI SINI LOKASINYA (20000 = 20ms)
        },
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    print("? Camera Started (Anti-Blur Mode)")
except Exception as e:
    print(f"? Camera Error: {e}")


app = Flask(__name__)


# =======================
# GLOBAL VARIABLES
# =======================
curr_x, curr_y = 90, 90
step, margin = 0.5, 50      # Faster step for tracking
timeout_sec = 7.0        
is_speaking = False
frame_count = 0
global_frame_jpg = None
frame_lock = threading.Lock()


# Manual Override Flag
is_manual_override = False
override_timer = 0


# Auto Mode State
is_auto_mode = False
search_start_time = 0
search_phase = "IDLE"


# Serial Lock
serial_lock = threading.Lock()


# Status Text
system_status = "READY"


# Last Known Box
last_box = None
last_box_time = 0


# Sweep Variables
sweep_angle = 90
sweep_dir = 1


# =======================
# HELPERS
# =======================
def robot_speak(text):
   """Non-blocking Text-to-Speech"""
   def _speak():
       global is_speaking
       is_speaking = True
       print(f"🤖 Robot: {text}")
       try:
           tts = gTTS(text=text, lang='id')
           tts.save("/tmp/response.mp3")
           pygame.mixer.init()
           pygame.mixer.music.load("/tmp/response.mp3")
           pygame.mixer.music.play()
           while pygame.mixer.music.get_busy(): continue
       except: pass
       is_speaking = False
   threading.Thread(target=_speak).start()


def send_uart_command(action, duration=0):
   """Sends command to ESP32 safely using the Lock"""
   global is_manual_override, override_timer, system_status, is_auto_mode, search_phase, search_start_time
  
   # If sending manual command, disable Auto Mode
   if action not in ['T']:
       is_auto_mode = False
       search_phase = "IDLE"
       is_manual_override = True
       override_timer = time.time() + (duration if duration > 0 else 2.0)
  
   if action == 'T':
       is_auto_mode = True
       search_phase = "MOVE"
       search_start_time = time.time()
       system_status = "AUTO SEARCH"
   elif action == 'S':
       system_status = "STOPPED"
   else:
       system_status = f"MANUAL: {action}"


   if ser:
       try:
           with serial_lock:
               command_str = f"{action}\n"
               ser.write(command_str.encode('utf-8'))
               print(f"⚡ Sent to ESP32: {command_str.strip()}")
          
           if duration > 0:
               def delayed_stop():
                   time.sleep(duration)
                   with serial_lock:
                       ser.write(b'S\n')
                       print("⚡ Auto-Stop Sent")
               threading.Thread(target=delayed_stop).start()
       except Exception as e:
           print(f"❌ Serial Error: {e}")


# =======================
# WEB INTERFACE (HTML)
# =======================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
   <meta charset="UTF-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
   <title>Hybrid Athlete Robot</title>
   <style>
       :root { --primary: #00ff88; --bg: #121212; --panel: #1e1e1e; }
       body { font-family: 'Courier New', monospace; background: var(--bg); color: var(--primary); text-align: center; margin: 0; padding: 10px; overscroll-behavior: none; }
       h1 { margin: 10px 0; font-size: 1.5rem; text-shadow: 0 0 10px var(--primary); }
       .cam-container { position: relative; width: 100%; max-width: 480px; margin: 0 auto; border: 2px solid var(--primary); border-radius: 10px; overflow: hidden; background: #000; }
       .cam-feed { width: 100%; height: auto; display: block; }
       .status-box { margin: 10px auto; padding: 10px; width: 90%; max-width: 460px; background: var(--panel); border: 1px solid #333; border-radius: 5px; color: #fff; font-size: 0.9rem; min-height: 40px; }
       .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 300px; margin: 20px auto; }
       .btn { background: var(--panel); border: 1px solid var(--primary); color: var(--primary); padding: 15px 0; font-size: 1.5rem; border-radius: 10px; cursor: pointer; touch-action: manipulation; user-select: none; transition: background 0.1s, transform 0.1s; }
       .btn:active { background: var(--primary); color: #000; transform: scale(0.95); }
       .btn-mic { grid-column: 1 / -1; background: #0044ff; color: white; border: none; display: flex; align-items: center; justify-content: center; gap: 10px; }
       .btn-mic.recording { background: #ff0044; animation: pulse 1s infinite; }
       .btn-auto { grid-column: 1 / -1; background: #ffaa00; color: #000; font-weight: bold; font-size: 1.2rem; }
       @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
   </style>
</head>
<body>
   <h1>COMMAND CENTER</h1>
   <div class="cam-container"><img class="cam-feed" src="/video_feed"></div>
   <div id="status" class="status-box">System Ready.</div>
   <div class="controls">
       <button id="micBtn" class="btn btn-mic" onclick="toggleMic()"><span>🎤</span> <span id="micText">VOICE COMMAND</span></button>
       <div></div><button class="btn" onpointerdown="send('F')" onpointerup="send('S')">▲</button><div></div>
       <button class="btn" onpointerdown="send('L')" onpointerup="send('S')">◄</button><button class="btn" onclick="send('S')">■</button><button class="btn" onpointerdown="send('R')" onpointerup="send('S')">►</button>
       <div></div><button class="btn" onpointerdown="send('B')" onpointerup="send('S')">▼</button><div></div>
       <button class="btn btn-auto" onclick="send('T')">♻️ CARI SAMPAH</button>
   </div>
   <script>
       function send(action) {
           fetch('/manual_cmd', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cmd: action}) });
           if(action === 'T') document.getElementById('status').innerText = "MODE: AUTO SEARCH";
           else if(action !== 'S') document.getElementById('status').innerText = "Manual: " + action;
       }
       let isRecording = false; let recognition;
       if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
           const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
           recognition = new SpeechRecognition(); recognition.continuous = false; recognition.lang = 'id-ID';
           recognition.onstart = function() { isRecording = true; document.getElementById('micBtn').classList.add('recording'); document.getElementById('micText').innerText = "LISTENING..."; };
           recognition.onend = function() { isRecording = false; document.getElementById('micBtn').classList.remove('recording'); document.getElementById('micText').innerText = "VOICE COMMAND"; };
           recognition.onresult = function(event) {
               const transcript = event.results[0][0].transcript; document.getElementById('status').innerText = "Voice: " + transcript;
               fetch('/process_voice', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: transcript}) });
           };
       }
       function toggleMic() { if (isRecording) recognition.stop(); else recognition.start(); }
   </script>
</body>
</html>
"""


# =======================
# FLASK ROUTES
# =======================
@app.route('/')
def index():
   return render_template_string(HTML_PAGE)


@app.route('/manual_cmd', methods=['POST'])
def manual_cmd():
   data = request.json
   cmd = data.get('cmd', '')
   if cmd:
       send_uart_command(cmd)
       return jsonify({"status": "sent", "cmd": cmd})
   return jsonify({"status": "error"})


@app.route('/process_voice', methods=['POST'])
def web_voice_input():
   global system_status
   data = request.json
   text = data.get('text', '').lower()
   print(f"🗣️ Voice: {text}")
  
   system_status = f"VOICE: {text[:15]}..."
  
   if "stop" in text or "berhenti" in text:
       send_uart_command('S')
       robot_speak("Berhenti.")
   elif "maju" in text:
       robot_speak("Maju.")
       send_uart_command('F')
   elif "mundur" in text:
       robot_speak("Mundur.")
       send_uart_command('B')
   elif "putar" in text or "kanan" in text:
       robot_speak("Kanan.")
       send_uart_command('R', duration=1.0)
   elif "kiri" in text:
       robot_speak("Kiri.")
       send_uart_command('L', duration=1.0)
   elif "sampah" in text:
       robot_speak("Cari sampah.")
       send_uart_command('T')
   else:
       robot_speak("Tidak kenal.")
      
   return jsonify({"status": "processed"})


@app.route('/video_feed')
def video_feed():
   def generate():
       while True:
           with frame_lock:
               if global_frame_jpg is None: continue
               data = global_frame_jpg
           yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
           time.sleep(0.05)
   return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =======================
# VISION THREAD
# =======================
def ai_logic_loop():
   global curr_x, curr_y, global_frame_jpg, frame_count, is_manual_override, system_status, last_box, last_box_time, is_auto_mode, search_phase, search_start_time, sweep_angle, sweep_dir
   print("🧠 AI Vision Thread Started...")
   last_seen_time = time.time()
  
   while True:
       try:
           try:
               frame_rgb = picam2.capture_array()
           except:
               time.sleep(0.1); continue


           frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
           rows, cols, _ = frame_bgr.shape
           center_x = cols // 2
           center_y = rows // 2
          
           frame_count += 1
          
           # --- DETECTION LOGIC ---
           if not is_manual_override:
              
               # [OPTIMIZATION] Run heavy AI every N frames
               if model and (frame_count % (SKIP_FRAMES + 1) == 0):
                   results = model(frame_bgr, imgsz=192, stream=True, verbose=False, conf=CONF_THRESHOLD)
                   found_target = False
                   beep_trigger = 0


                   for r in results:
                       boxes = r.boxes
                       for box in boxes:
                           if int(box.cls[0]) in [39, 41]: # Bottle/Cup
                               found_target = True
                               last_seen_time = time.time()
                               x1, y1, x2, y2 = box.xyxy[0]
                               last_box = (int(x1), int(y1), int(x2), int(y2))
                               last_box_time = time.time()
                               if float(box.conf[0]) > 0.7: beep_trigger = 1
                               break
                       if found_target: break
                  
                   if not found_target and time.time() - last_box_time > 0.5:
                       last_box = None


               # --- TRACKING / SEARCH LOGIC ---
               if last_box:
                   # [STATE: TARGET LOCKED]
                   x1, y1, x2, y2 = last_box
                   obj_x = int((x1 + x2) / 2)
                   obj_y = int((y1 + y2) / 2)
                  
                   # Visuals
                   cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                   cv2.circle(frame_bgr, (obj_x, obj_y), 5, (0, 0, 255), -1)
                   cv2.putText(frame_bgr, "TRASH", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                  
                   # [RESTORED] Tracking Calculation (Updates curr_x/y)
                   if obj_x < center_x - margin: curr_x += step
                   elif obj_x > center_x + margin: curr_x -= step
                   if obj_y < center_y - margin: curr_y -= step
                   elif obj_y > center_y + margin: curr_y += step
                  
                   curr_x = max(0, min(180, curr_x))
                   curr_y = max(85, min(95, curr_y)) # Keep head mostly level
                  
                   system_status = "LOCKED ON TARGET"
                   search_start_time = time.time()


               elif is_auto_mode:
                   # [STATE: AUTO SEARCHING]
                   elapsed = time.time() - search_start_time
                  
                   if elapsed < 12:
                       # PHASE 1: MOVE
                       if search_phase != "MOVE":
                           if ser:
                               with serial_lock: ser.write(b'T\n')
                           search_phase = "MOVE"
                       curr_x = 90
                       system_status = f"SEARCH: MOVE ({int(12-elapsed)}s)"
                       cv2.putText(frame_bgr, "MOVING...", (50, rows - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                  
                   elif elapsed < 20:
                       # PHASE 2: SWEEP
                       if search_phase != "SWEEP":
                           if ser:
                               with serial_lock: ser.write(b'S\n')
                           search_phase = "SWEEP"
                      
                       system_status = f"SEARCH: SWEEP ({int(20-elapsed)}s)"
                       cv2.putText(frame_bgr, "SWEEPING...", (50, rows - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                      
                       # Sweep Logic
                       sweep_angle += (2 * sweep_dir)
                       if sweep_angle > 120: sweep_dir = -1
                       if sweep_angle < 60: sweep_dir = 1
                       curr_x = sweep_angle
                  
                   else:
                       # RESET CYCLE
                       search_start_time = time.time()
                       search_phase = "IDLE"
              
               else:
                   # [STATE: MANUAL/IDLE]
                   if time.time() - last_seen_time > timeout_sec:
                       curr_x = 90
                       cv2.putText(frame_bgr, "SCANNING...", (50, rows - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)


               # Send Data to ESP32
               if ser:
                   with serial_lock:
                       msg = f"{curr_x},{curr_y},{beep_trigger}\n"
                       ser.write(msg.encode('utf-8'))


           else:
               # Manual Mode Visuals
               if time.time() > override_timer:
                   is_manual_override = False
                   system_status = "TRACKING ACTIVE"
               else:
                   cv2.putText(frame_bgr, "MANUAL OVERRIDE", (10, rows - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


           cv2.putText(frame_bgr, system_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


           ret, buffer = cv2.imencode('.jpg', frame_bgr)
           if ret:
               with frame_lock: global_frame_jpg = buffer.tobytes()


       except Exception as e:
           time.sleep(0.01)


if __name__ == '__main__':
   print("🚀 STARTING HYBRID SYSTEM...")
  
   # Handle clean exit
   def signal_handler(sig, frame):
       print("\n🛑 Shutting down...")
       if picam2: picam2.stop()
       if ser: ser.close()
       sys.exit(0)
   signal.signal(signal.SIGINT, signal_handler)


   vision_thread = threading.Thread(target=ai_logic_loop)
   vision_thread.daemon = True
   vision_thread.start()
  
   # Run Flask
   app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
a

