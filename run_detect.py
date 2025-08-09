# run_detect.py (robust)
import os, math
import cv2
from ultralytics import YOLO

IN_PATH  = "input_videos/clip.mp4"         # 改成你的文件名
OUT_DIR  = "output_videos"
OUT_MP4  = os.path.join(OUT_DIR, "detect_only.mp4")
OUT_AVI  = os.path.join(OUT_DIR, "detect_only.avi")

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(IN_PATH)
if not cap.isOpened():
    raise SystemExit(f"❌ 无法打开视频：{IN_PATH}")

# 先读一帧，拿到真实尺寸（有些容器里元数据不准）
ok, frame = cap.read()
if not ok:
    raise SystemExit("❌ 读取首帧失败")

h, w = frame.shape[:2]

# 兜底 FPS（很多剪辑过的片段在 Windows 下读出来是 0）
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or math.isnan(fps) or fps < 1:
    fps = 25.0  # 给个安全值
    print(f"⚠️ 原视频 FPS 异常，使用兜底 FPS={fps}")

# 模型
model = YOLO("yolov8n.pt")

def write_video(path, fourcc_str):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    return vw

# 先尝试 mp4v -> mp4
out = write_video(OUT_MP4, "mp4v")
if not out.isOpened():
    print("⚠️ mp4v 打开失败，回退到 XVID/AVI（兼容性更好）")
    out = write_video(OUT_AVI, "XVID")
    if not out.isOpened():
        raise SystemExit("❌ 无法创建任何视频输出（mp4/avi 都失败）")

# 处理首帧
res = model(frame, conf=0.25)[0]
for b in res.boxes:
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    cls = int(b.cls[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, model.model.names[cls], (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
out.write(frame)

# 剩余帧循环
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model(frame, conf=0.25)[0]
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, model.model.names[cls], (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
print("✅ 输出：", OUT_AVI if OUT_AVI.endswith(".avi") and os.path.exists(OUT_AVI) else OUT_MP4)
