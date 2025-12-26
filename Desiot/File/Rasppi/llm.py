from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import serial
import time
import numpy as np
import threading
import speech_recognition as sr
from gtts import gTTS
import pygame
import google.generativeai as genai
import json
import os

# =======================
# CONFIGURATION
# =======================
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200
API_KEY = "AIzaSyBgtlYIYFjyqZlLkFtB0QK0oTTAVIUdauo" # Replace with your actual key

# Configure AI
genai.configure(api_key=API_KEY)
model_llm = genai.GenerativeModel('gemini-1.5-flash')

# Global Variables
curr_x, curr_y = 90, 90
step, margin = 2, 60
timeout_sec = 5.0
is_speaking = False
is_processing_command = False

# Setup Hardware
print("Loading YOLO Model...")
model = YOLO('yolov8n.pt')

try:
    ser = serial.Serial(SERIAL_PORT, baudrate=BAUDRATE, timeout=0.1)
    print(f"Connected to ESP32 on {SERIAL_PORT}")
except:
    print("WARNING: ESP32 NOT CONNECTED (Simulation Mode)")
    ser = None

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

# =======================
# AUDIO FUNCTIONS
# =======================
def robot_speak(text):
    global is_speaking
    is_speaking = True
    print(f"🤖 Robot: {text}")
    try:
        tts = gTTS(text=text, lang='id') # Indonesian accent
        tts.save("response.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    except:
        pass
    is_speaking = False

def process_voice_command(user_text):
    global is_processing_command
    is_processing_command = True
    
    print(f"User Command: {user_text}")

    # 1. EMERGENCY STOP (Fast Check)
    if "stop" in user_text.lower() or "berhenti" in user_text.lower():
        if ser: ser.write(b'S')
        robot_speak("Stopping immediately.")
        is_processing_command = False
        return

    # 2. ASK GEMINI TO CONVERT TEXT -> JSON
    prompt = f"""
    You are a robot controller. Analyze this Indonesian command: "{user_text}"
    
    If it is a movement command, output JSON:
    {{"type": "move", "action": "ACTION_CODE", "duration": SECONDS}}
    
    Action Codes:
    - Forward/Maju = F
    - Backward/Mundur = B
    - Left/Kiri = L
    - Right/Kanan = R
    
    If duration is not mentioned, default to 1.0.
    
    If it is a chat (e.g. "Halo"), output JSON:
    {{"type": "chat", "response": "Your reply in Indonesian"}}
    
    Output ONLY JSON. No markdown.
    """
    
    try:
        response = model_llm.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        if data["type"] == "move":
            action = data["action"]
            duration = float(data["duration"])
            
            robot_speak(f"Oke, bergerak {duration} detik.")
            
            # Send Command to ESP32
            if ser: ser.write(action.encode()) 
            time.sleep(duration)
            if ser: ser.write(b'S') # Auto Stop
            
        elif data["type"] == "chat":
            robot_speak(data["response"])
            
    except Exception as e:
        print(f"AI Error: {e}")
        robot_speak("Maaf, saya tidak mengerti.")

    is_processing_command = False

# =======================
# VOICE LISTENER THREAD
# =======================
def background_listener():
    recognizer = sr.Recognizer()
    mic = sr.Microphone() # Use specific index if needed: sr.Microphone(device_index=X)
    
    print("🎤 Voice Listening Active... Say 'Robot' to start.")
    
    while True:
        if is_speaking or is_processing_command:
            time.sleep(0.5)
            continue
            
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
            
            text = recognizer.recognize_google(audio, language="id-ID").lower()
            
            if "robot" in text:
                command = text.replace("robot", "").strip()
                if command:
                    process_voice_command(command)
                else:
                    robot_speak("Ya?")
                    
        except:
            pass 

# Start Voice in Background
t = threading.Thread(target=background_listener)
t.daemon = True
t.start()

# =======================
# MAIN LOOP
# =======================
last_seen_time = time.time()

print("SYSTEM READY. Press 'q' to quit.")

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        rows, cols, _ = frame_bgr.shape
        center_x = cols // 2
        center_y = rows // 2

        # Only run tracking if NOT processing a voice command
        if not is_processing_command:
            results = model(frame_bgr, imgsz=192, stream=True, verbose=False, conf=0.5)
            found_target = False
            beep_trigger = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    # 39=Bottle, 41=Cup
                    if cls_id == 39 or cls_id == 41:
                        found_target = True
                        last_seen_time = time.time()
                        
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        obj_x = int((x1 + x2) / 2)
                        obj_y = int((y1 + y2) / 2)

                        # Visuals
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame_bgr, (obj_x, obj_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame_bgr, "TRASH", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                        # --- TRACKING LOGIC ---
                        if obj_x < center_x - margin: curr_x += step
                        elif obj_x > center_x + margin: curr_x -= step
                        
                        if obj_y < center_y - margin: curr_y -= step
                        elif obj_y > center_y + margin: curr_y += step

                        curr_x = max(0, min(180, curr_x))
                        curr_y = max(70, min(120, curr_y))

                        if float(box.conf[0]) > 0.8: beep_trigger = 1
                        break 
                if found_target: break

            # Timeout Reset
            if not found_target and (time.time() - last_seen_time > timeout_sec):
                curr_x = 90
                curr_y = 90
                cv2.putText(frame_bgr, "SCANNING...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Send to ESP32
            if ser:
                msg = f"{curr_x},{curr_y},{beep_trigger}\n"
                ser.write(msg.encode('utf-8'))

        # Display Status
        status = "LISTENING..." if not is_processing_command else "THINKING..."
        cv2.putText(frame_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        cv2.imshow("Smart Robot", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    if ser: ser.close()
    cv2.destroyAllWindows()
