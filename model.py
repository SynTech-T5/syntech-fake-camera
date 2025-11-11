# ============================================================
# YOLO Pose Detection (Optimized: CPU Decode + YOLO11n GPU Infer + GPU Encode)
# ============================================================

import os
import sys
import cv2
import torch
import time
import subprocess
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta

# ------------------------------------------------------------
# 🌍 Environment Config
# ------------------------------------------------------------
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

target_cam = sys.argv[1] if len(sys.argv) > 1 else "city-traffic"

# ✅ ใช้โมเดล YOLO11n เดิม
model_path = "/app/models/yolo11n-pose.pt"
conf_threshold = 0.3
cooldown_seconds = 1

# ✅ ลดความละเอียด + skip frame เพื่อความลื่น
width, height, fps = 1280, 720, 25
process_every = 2  # 🔹 skip ทุก 2 เฟรม ลดโหลด GPU

# ------------------------------------------------------------
# ⚙️ Load YOLO model (GPU if available)
# ------------------------------------------------------------
model = YOLO(model_path)
if torch.cuda.is_available():
    model.to("cuda")
    print("⚙️ Using GPU for YOLO inference")
else:
    model.to("cpu")
    print("⚙️ Using CPU fallback for YOLO")

# ------------------------------------------------------------
# 🎥 Input: FFmpeg CPU Decode (เพิ่ม threads + ลด buffer)
# ------------------------------------------------------------
input_url = f"rtsp://viewer:viewpass@rtsp-server:8003/{target_cam}"
print(f"📡 Input stream: {input_url}")

ffmpeg_input = [
    "ffmpeg",
    "-hwaccel", "none",        # ✅ CPU Decode
    "-threads", "2",           # ใช้ multi-thread decode
    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-rtbufsize", "64M",
    "-r", "25",                # ✅ บังคับ frame rate 25 fps
    "-vsync", "2",             # ✅ ปรับการ sync เฟรมให้เสถียร (drop/duplicate minimal)
    "-i", input_url,
    "-pix_fmt", "bgr24",
    "-f", "rawvideo",
    "-vf", f"scale={width}:{height},format=bgr24",
    "pipe:1",
]

# ------------------------------------------------------------
# 🚀 Output: FFmpeg GPU Encode (NVENC, compatible FFmpeg 4.4.2)
# ------------------------------------------------------------
output_cam = f"{target_cam}-processed"
output_url = f"rtsp://pub:pubpass@rtsp-server:8003/{output_cam}"
print(f"🚀 Output stream: {output_url}")

ffmpeg_output = [
    "ffmpeg",
    "-loglevel", "error",
    "-y",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}",
    "-i", "-",
    "-c:v", "h264_nvenc",
    "-preset", "p5",
    "-tune", "ull",  # ✅ ultra-low latency (สำคัญ)
    "-profile:v", "baseline",  # ✅ บังคับใช้ Baseline Profile
    "-level", "4.1",
    "-b:v", "2M",
    "-g", "25",
    "-bf", "0",                # ✅ ปิด B-frames (WebRTC ต้องการ)
    "-pix_fmt", "yuv420p",     # ✅ ให้เป็น pixel format มาตรฐาน
    "-rtsp_transport", "tcp",
    "-fflags", "flush_packets",
    "-f", "rtsp",
    output_url
]

# ------------------------------------------------------------
# 🔁 Start pipelines
# ------------------------------------------------------------
input_proc = subprocess.Popen(ffmpeg_input, stdout=subprocess.PIPE, bufsize=10**8)
output_proc = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE)

frame_size = width * height * 3
frame_count = 0
start_time = time.time()
last_log_time = datetime.min

print("✅ Starting YOLO Pose Detection (YOLO11n, optimized for real-time)...")

try:
    while True:
        raw_frame = input_proc.stdout.read(frame_size)
        if not raw_frame:
            print("⚠️ Input stream ended or lost, reconnecting...")
            time.sleep(3)
            input_proc.terminate()
            input_proc = subprocess.Popen(ffmpeg_input, stdout=subprocess.PIPE, bufsize=10**8)
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        frame_count += 1

        # ✅ skip frame เพื่อลดโหลด GPU
        if frame_count % process_every != 0:
            try:
                output_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                print("⚠️ Output stream broken, restarting encoder...")
                output_proc.terminate()
                output_proc = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE)
            continue

        # 🧠 YOLO Pose Detection
        start_infer = time.time()
        results = model.predict(frame, task="pose", conf=conf_threshold, imgsz=480, verbose=False)
        infer_time = (time.time() - start_infer) * 1000
        annotated = results[0].plot()

        try:
            output_proc.stdin.write(annotated.tobytes())
        except (BrokenPipeError, IOError):
            print("⚠️ Output stream broken, restarting encoder...")
            output_proc.terminate()
            output_proc = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE)
            continue

        # 👥 Log people count (cooldown 5 วินาที)
        if results and results[0].keypoints is not None:
            num_people = len(results[0].keypoints.xy)
            now = datetime.now()
            if num_people > 0 and now - last_log_time >= timedelta(seconds=cooldown_seconds):
                print(f"[{target_cam}] 👥 {num_people} person(s) | Infer: {infer_time:.1f} ms | {now.strftime('%H:%M:%S')}")
                last_log_time = now

        # 🎞️ FPS monitoring ทุก 5 วินาที
        if time.time() - start_time >= 5:
            fps_calc = frame_count / (time.time() - start_time)
            print(f"🎞️ FPS: {fps_calc:.2f}")
            frame_count = 0
            start_time = time.time()

except KeyboardInterrupt:
    print("🛑 Interrupted by user")

finally:
    input_proc.terminate()
    output_proc.stdin.close()
    output_proc.wait()
    print("🧹 Cleanup complete.")
