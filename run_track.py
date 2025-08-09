# run_track.py  —— 兼容版（不依赖 ColorPalette）
import os, math, time
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

IN_PATH  = "input_videos/clip.mp4"   # 改成你的文件
OUT_DIR  = "output_videos"
OUT_MP4  = os.path.join(OUT_DIR, "track.mp4")
OUT_AVI  = os.path.join(OUT_DIR, "track.avi")

CONF_TH = 0.25
IMGSZ   = 640
TRACE_LEN = 50

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(IN_PATH)
if not cap.isOpened():
    raise SystemExit(f"❌ 无法打开视频：{IN_PATH}")

ok, frame0 = cap.read()
if not ok:
    raise SystemExit("❌ 读取首帧失败")

h, w = frame0.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or math.isnan(fps) or fps < 1:
    fps = 25.0
    print(f"⚠️ 原视频 FPS 异常，使用兜底 FPS={fps}")

model = YOLO("yolov8n.pt")
try:
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
        print("✅ 使用 GPU 推理")
except Exception:
    pass

# 兼容不同 Supervision 版本的 ByteTrack 构造
try:
    tracker = sv.ByteTrack()
except Exception:
    try:
        from supervision.tracker.byte_tracker import ByteTrack
        tracker = ByteTrack()
    except Exception as e:
        raise SystemExit(f"❌ 无法创建 ByteTrack：{e}")

trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=TRACE_LEN)

def open_writer():
    vw = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if vw.isOpened():
        return vw, OUT_MP4
    print("⚠️ mp4v 打开失败，回退 XVID/AVI")
    vw = cv2.VideoWriter(OUT_AVI, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if vw.isOpened():
        return vw, OUT_AVI
    raise SystemExit("❌ 无法创建视频输出（mp4/avi 都失败）")

out, out_path = open_writer()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def color_for_id(tid: int):
    # 用“黄金比例”扰动的 HSV 生成稳定可区分颜色，再转 BGR
    import colorsys
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 1.0)
    return int(b*255), int(g*255), int(r*255)  # BGR

t0 = time.time()
frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model(frame, conf=CONF_TH, imgsz=IMGSZ)[0]
    det = sv.Detections.from_ultralytics(res)

    keep_idx = [i for i, c in enumerate(det.class_id) if c in (0, 32)]
    det = det[keep_idx]

    person_mask = det.class_id == 0
    ball_mask   = det.class_id == 32
    persons = det[person_mask]
    balls   = det[ball_mask]

    tracked = tracker.update_with_detections(persons)

    # 先画轨迹（用默认颜色即可）
    frame = trace_annotator.annotate(scene=frame, detections=tracked)

    # 用 OpenCV 画框 + 标签（每个 ID 自定义颜色）
    for i in range(len(tracked)):
        x1, y1, x2, y2 = map(int, tracked.xyxy[i])
        tid = int(tracked.tracker_id[i])
        col = color_for_id(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # 足球：取最高置信度的一个画圆
    if len(balls):
        idx = int(np.argmax(balls.confidence))
        x1, y1, x2, y2 = map(int, balls.xyxy[idx])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 215, 255), -1)
        cv2.putText(frame, "ball", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    out.write(frame)
    frames += 1

cap.release()
out.release()
dt = time.time() - t0
print(f"✅ 输出：{out_path} | 帧数={frames} | 用时={dt:.1f}s | 平均 {frames/max(dt,1e-6):.1f} FPS")
