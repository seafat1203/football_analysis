# run_possession.py —— 控球归属 + 占有率 (兼容版)
import os, math, time, csv
import cv2
import numpy as np
from collections import deque, Counter, defaultdict
from ultralytics import YOLO
import supervision as sv

IN_PATH   = "input_videos/clip.mp4"      # 改成你的视频
OUT_DIR   = "output_videos"
OUT_MP4   = os.path.join(OUT_DIR, "possession.mp4")
OUT_AVI   = os.path.join(OUT_DIR, "possession.avi")
CSV_PATH  = os.path.join(OUT_DIR, "possession_stats.csv")

CONF_TH   = 0.25     # 检测阈值
IMGSZ     = 640      # 推理分辨率
SMOOTH_S  = 2.0      # 滑窗秒数（用于平滑控球归属）
TRACE_LEN = 50       # 轨迹长度

os.makedirs(OUT_DIR, exist_ok=True)

# --- 打开视频 & FPS 兜底 ---
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
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- 模型 & 设备 ---
model = YOLO("yolov8n.pt")
try:
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
        print("✅ 使用 GPU 推理")
except Exception:
    pass

# --- Tracker（兼容不同 supervision 版本） ---
try:
    tracker = sv.ByteTrack()
except Exception:
    try:
        from supervision.tracker.byte_tracker import ByteTrack
        tracker = ByteTrack()
    except Exception as e:
        raise SystemExit(f"❌ 无法创建 ByteTrack：{e}")

trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=TRACE_LEN)

# --- 视频写入（先 mp4v，失败回退 avi/xvid） ---
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

# --- 工具：ID 颜色、中心点 ---
def color_for_id(tid: int):
    import colorsys
    h_ = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h_, 0.7, 1.0)
    return int(b*255), int(g*255), int(r*255)  # BGR

def centers_from_xyxy(xyxy_array):
    # xyxy -> (cx, cy)
    c = []
    for x1,y1,x2,y2 in xyxy_array:
        c.append(((x1+x2)/2.0, (y1+y2)/2.0))
    return np.array(c, dtype=np.float32) if len(c) else np.zeros((0,2), dtype=np.float32)

# --- 统计结构 ---
win_len  = max(1, int(SMOOTH_S * fps))     # 滑窗帧数
window   = deque(maxlen=win_len)           # 记录最近 win_len 帧的“拥有者ID或None”
possess_frames = Counter()                 # ID -> 帧数
last_owner = None
frame_count = 0
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # YOLO 推理
    res = model(frame, conf=CONF_TH, imgsz=IMGSZ)[0]
    det = sv.Detections.from_ultralytics(res)

    # 只保留人(0)与球(32)
    keep_idx = [i for i, c in enumerate(det.class_id) if c in (0, 32)]
    det = det[keep_idx]

    person_mask = det.class_id == 0
    ball_mask   = det.class_id == 32
    persons = det[person_mask]
    balls   = det[ball_mask]

    # 跟踪（仅对人）
    tracked = tracker.update_with_detections(persons)

    # 画轨迹
    frame = trace_annotator.annotate(scene=frame, detections=tracked)

    # 人：画框与 ID
    for i in range(len(tracked)):
        x1,y1,x2,y2 = map(int, tracked.xyxy[i])
        tid = int(tracked.tracker_id[i])
        col = color_for_id(tid)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # 球：取最高置信度一个
    ball_center = None
    if len(balls):
        idx = int(np.argmax(balls.confidence))
        x1,y1,x2,y2 = map(int, balls.xyxy[idx])
        cx, cy = (x1+x2)//2, (y1+y2)//2
        ball_center = np.array([cx, cy], dtype=np.float32)
        cv2.circle(frame, (cx, cy), 6, (0, 215, 255), -1)
        cv2.putText(frame, "ball", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    # —— 控球归属：找“离球最近”的球员 ID（无球时记 None） —— #
    owner = None
    if ball_center is not None and len(tracked) > 0:
        player_centers = centers_from_xyxy(tracked.xyxy)
        dists = np.linalg.norm(player_centers - ball_center, axis=1)
        j = int(np.argmin(dists))
        owner = int(tracked.tracker_id[j])

        # 在视频上标注控球者
        ox1, oy1, ox2, oy2 = map(int, tracked.xyxy[j])
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 255), 3)
        cv2.putText(frame, f"OWNER ID {owner}", (ox1, max(0, oy1-24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # —— 平滑：滑窗众数 —— #
    window.append(owner)
    if len(window) == window.maxlen:
        vals = [x for x in window if x is not None]
        if len(vals) > 0:
            # 众数作为当前真实拥有者
            cur_owner = Counter(vals).most_common(1)[0][0]
            possess_frames[cur_owner] += 1  # 每满一个窗口，+1 帧（近似）
            last_owner = cur_owner
        else:
            last_owner = None

    # HUD：左上角显示当前拥有者
    hud_text = f"Owner: {last_owner if last_owner is not None else '-'}"
    cv2.putText(frame, hud_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 50), 2)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# —— 汇总统计（换算为秒 & 占比）—— #
total_frames_for_stats = sum(possess_frames.values())
total_secs = total_frames_for_stats / max(fps, 1e-6)
stats = []
for pid, fcnt in sorted(possess_frames.items()):
    secs = fcnt / max(fps, 1e-6)
    ratio = (secs / total_secs * 100.0) if total_secs > 0 else 0.0
    stats.append((pid, round(secs, 2), round(ratio, 1)))

# 保存 CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    wtr = csv.writer(f)
    wtr.writerow(["player_id", "possession_seconds", "possession_percent"])
    for row in stats:
        wtr.writerow(row)

dt = time.time() - t0
print(f"✅ 输出视频：{out_path}")
print(f"✅ 统计CSV：{CSV_PATH}")
print(f"帧数={frame_count} | 用时={dt:.1f}s | 平均 {frame_count/max(dt,1e-6):.1f} FPS")
print("控球统计（按ID）：", stats)
