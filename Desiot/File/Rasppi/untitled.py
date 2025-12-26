from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import serial
import time
import numpy as np

# =======================
# CONFIGURATION
# =======================
SERIAL_PORT = "/dev/serial0" 
BAUDRATE = 115200

# Tuning
curr_x = 90
curr_y = 90
step = 2        
margin = 60     
timeout_sec = 5.0

# Load Model
print("Loading Model... Please wait.")
model = YOLO('yolov8n.pt') 

# Setup Serial
try:
    ser = serial.Serial(SERIAL_PORT, baudrate=BAUDRATE, timeout=0.1)
    print(f"Connected to ESP32 on {SERIAL_PORT}")
except:
    print("WARNING: ESP32 NOT CONNECTED (Simulation Mode)")
    ser = None

# Setup Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

last_seen_time = time.time() 

print("SYSTEM READY. Press 'q' to quit.")

try:
    while True:
        # 1. Capture
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        rows, cols, _ = frame_bgr.shape
        center_x = cols // 2
        center_y = rows // 2

        # 2. Fast Inference
        # imgsz=192 makes it MUCH faster
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
                    
                    # Y-Axis (Inverted Logic)
                    if obj_y < center_y - margin: curr_y -= step
                    elif obj_y > center_y + margin: curr_y += step

                    # Limits
                    curr_x = max(0, min(180, curr_x))
                    curr_y = max(70, min(120, curr_y))

                    if float(box.conf[0]) > 0.8: beep_trigger = 1
                    break 
            if found_target: break

        # 3. Timeout Reset
        if not found_target and (time.time() - last_seen_time > timeout_sec):
            curr_x = 90
            curr_y = 90
            cv2.putText(frame_bgr, "SCANNING...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # 4. Send to ESP32
        if ser:
            msg = f"{curr_x},{curr_y},{beep_trigger}\n"
            ser.write(msg.encode('utf-8'))

        # 5. Display on Laptop/Monitor
        cv2.putText(frame_bgr, f"Servo: {curr_x},{curr_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Trash Detection", frame_bgr)
        
        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    if ser: ser.close()
    cv2.destroyAllWindows()
