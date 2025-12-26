from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time
import serial  # <--- UART

# =======================
# KONFIGURASI MODEL
# =======================
MODEL_PATH = "best.onnx"
CLASSES_FILE = "classes.txt"
INPUT_SIZE = 224  # YOLOv8 classification biasanya 224x224

# =======================
# KONFIGURASI UART
# =======================
SERIAL_PORT = "/dev/serial0"  # symlink ke ttyS0 di Pi-mu
BAUDRATE = 115200


# =======================
# LOAD KELAS & MODEL
# =======================
with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(f"Loaded {len(classes)} classes: {classes}")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model loaded successfully!")
print(f"Expected input shape: {session.get_inputs()[0].shape}")


# =======================
# FUNGSI PREPROCESS & INFERENCE
# =======================
def preprocess(image_bgr):
    """
    Preprocess image BGR untuk YOLOv8 Classification.
    Input: image BGR (OpenCV), bebas resolusi.
    Output: numpy array [1, 3, 224, 224]
    """
    # Resize ke 224x224 (sesuai training)
    img_resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE))

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalisasi ke [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    img_chw = img_normalized.transpose(2, 0, 1)

    # Tambah dimensi batch -> [1, 3, 224, 224]
    img_batch = np.expand_dims(img_chw, axis=0)

    return img_batch


def classify(image_bgr):
    """
    Klasifikasi image menggunakan ONNX model.
    Input: image BGR (OpenCV)
    Return: class_id, confidence, full_probabilities
    """
    input_tensor = preprocess(image_bgr)
    outputs = session.run([output_name], {input_name: input_tensor})

    predictions = outputs[0][0]  # [num_classes]
    class_id = int(np.argmax(predictions))
    confidence = float(predictions[class_id])

    return class_id, confidence, predictions


# =======================
# FUNGSI UI
# =======================
def draw_result_ui(frame_bgr, class_name, confidence, all_probs, fps):
    """
    Bangun layout UI:
    - Kamera di kiri
    - Panel info (kelas utama, top3, FPS, instruksi) di kanan
    """
    # Kanvas UI (bisa diubah sesuai selera)
    ui_h, ui_w = 480, 800
    ui = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
    ui[:] = (25, 25, 25)  # background gelap

    # ----- tampilkan kamera di kiri -----
    cam_target_w = 480
    h, w = frame_bgr.shape[:2]
    cam_target_h = int(h * cam_target_w / w)
    cam_view = cv2.resize(frame_bgr, (cam_target_w, cam_target_h))

    cam_x = 20
    cam_y = 40
    ui[cam_y:cam_y + cam_target_h, cam_x:cam_x + cam_target_w] = cam_view

    # ----- judul di atas -----
    title = "Garbage Classification"
    cv2.putText(
        ui,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # ----- panel info di kanan -----
    panel_x = cam_x + cam_target_w + 20
    panel_y = 60
    cv2.rectangle(
        ui,
        (panel_x - 10, panel_y - 30),
        (ui_w - 20, ui_h - 40),
        (40, 40, 40),
        -1,
    )

    # Kelas utama
    main_text = f"{class_name}: {confidence:.2%}"
    cv2.putText(
        ui,
        main_text,
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    panel_y += 40

    # Top-3 probabilitas
    top3_indices = np.argsort(all_probs)[-3:][::-1]
    for i, idx in enumerate(top3_indices):
        t = f"{i+1}. {classes[idx]}: {all_probs[idx]:.1%}"
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(
            ui,
            t,
            (panel_x, panel_y + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    panel_y += 3 * 30 + 20

    # FPS
    cv2.putText(
        ui,
        f"FPS: {fps}",
        (panel_x, ui_h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Instruksi kecil
    cv2.putText(
        ui,
        "q: quit",
        (panel_x, ui_h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        ui,
        "s: screenshot",
        (panel_x, ui_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    return ui


# =======================
# MAIN LOOP
# =======================
def main():
    # Inisialisasi kamera (Picamera2)
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1)  # waktu buat auto-exposure

    # Inisialisasi UART
    ser = serial.Serial(
        SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=0.1,
    )
    print(f"UART opened on {SERIAL_PORT} @ {BAUDRATE} baud")

    print("=" * 50)
    print("Garbage Classification System")
    print("=" * 50)
    print("Tekan 'q' untuk keluar")
    print("Tekan 's' untuk screenshot")
    print("=" * 50)

    cv2.namedWindow("Garbage Classification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Garbage Classification", 800, 480)

    fps_time = time.time()
    fps_counter = 0
    fps = 0
    screenshot_counter = 0

    last_class_name = "..."
    last_confidence = 0.0
    last_probs = np.zeros(len(classes), dtype=np.float32)

    try:
        while True:
            # Ambil frame dari kamera (RGB) -> BGR
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Jalankan klasifikasi
            class_id, confidence, all_probs = classify(frame_bgr)
            last_class_name = classes[class_id]
            last_confidence = confidence
            last_probs = all_probs

            # === Kirim hasil via UART ===
            # Format sederhana: hanya ID kelas + newline, misal "3\n"
            try:
                ser.write(f"{class_id}\n".encode("utf-8"))
            except Exception as e:
                print("UART write error:", e)

            # Hitung FPS tampilan
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            # Bangun UI frame komplit
            ui_frame = draw_result_ui(
                frame_bgr,
                last_class_name,
                last_confidence,
                last_probs,
                fps,
            )

            cv2.imshow("Garbage Classification", ui_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                screenshot_counter += 1
                filename = f"screenshot_{screenshot_counter}.jpg"
                cv2.imwrite(filename, ui_frame)
                print(f"Screenshot saved: {filename}")

    finally:
        picam2.stop()
        ser.close()
        cv2.destroyAllWindows()
        print("Program selesai!")


if __name__ == "__main__":
    main()
