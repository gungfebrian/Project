from picamera2 import Picamera2
import cv2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

for i in range(10):
    frame = picam2.capture_array()  # RGB
    print("Frame shape:", frame.shape)
    # OpenCV pakai BGR, jadi convert kalau mau diproses:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"frame_{i}.jpg", bgr)
    break  # ambil satu dulu

picam2.stop()
