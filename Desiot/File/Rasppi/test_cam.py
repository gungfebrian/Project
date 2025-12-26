import cv2

# Pakai V4L2 backend dan device /dev/video0
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

print("Opened:", cap.isOpened())
if not cap.isOpened():
    exit(1)

ret, frame = cap.read()
print("Got frame:", ret)

if ret:
    out_name = "frame_0.jpg"
    cv2.imwrite(out_name, frame)
    print("Saved:", out_name)

cap.release()
