"""
Phase 5 Demo — PPE Compliance Monitoring
========================================
Gradio app with 2 tabs:
  Tab 1 (Image): phase3_best.pt PPE detection overlay + per-image violation summary
  Tab 2 (Video): COCO yolov8n person detection + ByteTrack IDs + phase3_best.pt PPE
                 attributes + per-track temporal-voting compliance summary

Run:
  pip install ultralytics gradio opencv-python
  python ppe_demo_app.py
Then open http://127.0.0.1:7860

Put phase3_best.pt in the same folder (or change PPE_WEIGHTS below).
yolov8n.pt downloads automatically on first run.
"""

import os
import cv2
import numpy as np
import gradio as gr
from collections import defaultdict
from ultralytics import YOLO

# ----------------------------- config -----------------------------
PPE_WEIGHTS = "models/phase3_best.pt"      # your Phase 3 detector (10 classes)
PERSON_WEIGHTS = "models/yolov8n.pt"       # COCO model for person detection (CCTV domain gap fix)

PPE_CONF = 0.25                     # matches your 91.25% accuracy operating point
PERSON_CONF = 0.30
TARGET_FPS = 5                      # frames analysed per second of video (lower = faster, higher = smoother)
MAX_VIDEO_MINUTES = 0               # 0 = process the WHOLE video; e.g. 2 = only the first 2 minutes
VOTE_THRESHOLD = 0.30               # violation flagged if seen in >30% of a track's frames

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest", "NO-Gloves"}
POSITIVE_CLASSES = {"Hardhat", "Mask", "Safety Vest", "Gloves"}

# colors (BGR)
COL_VIOLATION = (0, 0, 255)
COL_OK = (0, 200, 0)
COL_PERSON = (255, 160, 0)
COL_OTHER = (200, 200, 0)

# ----------------------------- models -----------------------------
ppe_model = YOLO(PPE_WEIGHTS)


def box_color(name: str):
    if name in VIOLATION_CLASSES:
        return COL_VIOLATION
    if name in POSITIVE_CLASSES:
        return COL_OK
    if name.lower() == "person":
        return COL_PERSON
    return COL_OTHER


def draw_box(img, xyxy, label, color, thick=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def center_in_box(inner_xyxy, outer_xyxy, margin=0.15):
    """PPE box belongs to a person if its center lies inside the (slightly expanded) person box."""
    cx = (inner_xyxy[0] + inner_xyxy[2]) / 2
    cy = (inner_xyxy[1] + inner_xyxy[3]) / 2
    x1, y1, x2, y2 = outer_xyxy
    w, h = x2 - x1, y2 - y1
    return (x1 - margin * w <= cx <= x2 + margin * w and
            y1 - margin * h <= cy <= y2 + margin * h)


def mmss(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


# =========================== TAB 1: IMAGE ===========================
def run_image(image):
    if image is None:
        return None, "Upload an image first."
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    res = ppe_model(img, conf=PPE_CONF, verbose=False)[0]
    names = res.names
    counts = defaultdict(int)
    violations = []

    for b in res.boxes:
        cls_name = names[int(b.cls)]
        conf = float(b.conf)
        counts[cls_name] += 1
        draw_box(img, b.xyxy[0].tolist(), f"{cls_name} {conf:.2f}", box_color(cls_name))
        if cls_name in VIOLATION_CLASSES:
            violations.append(f"{cls_name} ({conf:.2f})")

    # summary text
    lines = ["## Detection summary"]
    if counts:
        for k in sorted(counts):
            lines.append(f"- **{k}**: {counts[k]}")
    else:
        lines.append("- No detections above confidence threshold.")

    lines.append("")
    if violations:
        lines.append(f"### ⚠️ VIOLATIONS DETECTED ({len(violations)})")
        for v in violations:
            lines.append(f"- 🔴 {v}")
    else:
        lines.append("### ✅ No PPE violations detected")

    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return out, "\n".join(lines)


# =========================== TAB 2: VIDEO ===========================
def run_video(video_path, progress=gr.Progress()):
    if video_path is None:
        return None, "Upload a video first."

    # fresh person model for every video, so track IDs start again from 1
    person_model = YOLO(PERSON_WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))          # only used for the progress bar
    max_frames = int(MAX_VIDEO_MINUTES * 60 * fps) if MAX_VIDEO_MINUTES > 0 else None
    if max_frames:
        total = min(total, max_frames) if total > 0 else max_frames
    stride = max(1, round(fps / TARGET_FPS))                 # e.g. 30 fps video -> every 6th frame

    out_path = os.path.join(os.path.dirname(video_path), "demo_output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (W, H))

    # per-track stats: track_id -> {frames_seen, violation_name -> count, ppe_name -> count}
    track_frames = defaultdict(int)
    track_attr = defaultdict(lambda: defaultdict(int))

    frame_idx = 0
    processed = 0
    while True:
        if max_frames and frame_idx >= max_frames:
            break
        if frame_idx % stride != 0:
            if not cap.grab():                               # skip frame without decoding (fast)
                break
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok:                                           # end of video
            break
        if total > 0:
            progress(min(frame_idx / total, 1.0),
                     desc=f"{mmss(frame_idx / fps)} / {mmss(total / fps)} of video")

        # 1) person tracking (COCO model + ByteTrack)
        tres = person_model.track(frame, persist=True, conf=PERSON_CONF,
                                  classes=[0], tracker="bytetrack.yaml",
                                  verbose=False)[0]
        persons = []  # (track_id, xyxy)
        if tres.boxes is not None and tres.boxes.id is not None:
            for b in tres.boxes:
                tid = int(b.id)
                xyxy = b.xyxy[0].tolist()
                persons.append((tid, xyxy))

        # 2) PPE attribute detection (phase3 model)
        pres = ppe_model(frame, conf=PPE_CONF, verbose=False)[0]
        names = pres.names
        ppe_boxes = []  # (name, conf, xyxy)
        for b in pres.boxes:
            n = names[int(b.cls)]
            if n.lower() == "person":      # skip: person handled by COCO model
                continue
            ppe_boxes.append((n, float(b.conf), b.xyxy[0].tolist()))

        # 3) associate PPE -> person, update votes
        for tid, pxyxy in persons:
            track_frames[tid] += 1
            for n, c, bxyxy in ppe_boxes:
                if n in VIOLATION_CLASSES | POSITIVE_CLASSES and center_in_box(bxyxy, pxyxy):
                    track_attr[tid][n] += 1

        # 4) draw
        for n, c, bxyxy in ppe_boxes:
            draw_box(frame, bxyxy, f"{n} {c:.2f}", box_color(n), 1)
        for tid, pxyxy in persons:
            # live status for this track so far
            vio = [v for v in VIOLATION_CLASSES
                   if track_attr[tid][v] / max(track_frames[tid], 1) > VOTE_THRESHOLD]
            col = COL_VIOLATION if vio else COL_PERSON
            draw_box(frame, pxyxy, f"ID {tid}" + (" NON-COMPLIANT" if vio else ""), col, 2)

        writer.write(frame)
        processed += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # ---- per-person compliance summary (temporal voting) ----
    lines = ["## Per-person compliance summary (temporal voting)",
             f"Video processed: **{mmss(frame_idx / fps)}** "
             f"({processed} frames analysed, every {stride} frame(s) at {fps:.0f} fps)",
             f"Tracks seen: **{len(track_frames)}** | vote threshold: {int(VOTE_THRESHOLD*100)}% of frames",
             "",
             "| Track ID | Frames | Violations (vote %) | Status |",
             "|---|---|---|---|"]
    for tid in sorted(track_frames):
        f = track_frames[tid]
        vios = []
        for v in sorted(VIOLATION_CLASSES):
            r = track_attr[tid][v] / f
            if r > VOTE_THRESHOLD:
                vios.append(f"{v} ({r:.0%})")
        status = "🔴 NON-COMPLIANT" if vios else "🟢 compliant"
        lines.append(f"| {tid} | {f} | {', '.join(vios) if vios else '—'} | {status} |")

    return out_path, "\n".join(lines)


# ============================== UI ==============================
with gr.Blocks(title="PPE Compliance Demo — Phase 5") as demo:
    gr.Markdown("# 🦺 PPE Compliance Monitoring — Phase 5 Demo\n"
                "Detection: custom YOLO11s (Phase 3) · Person tracking: COCO YOLOv8n + ByteTrack · "
                "Compliance: per-track temporal voting")

    with gr.Tab("Image — PPE Detection"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload image")
            img_out = gr.Image(label="Detections")
        img_summary = gr.Markdown()
        gr.Button("Run detection", variant="primary").click(
            run_image, inputs=img_in, outputs=[img_out, img_summary])

    with gr.Tab("Video — Tracking + Compliance"):
        with gr.Row():
            vid_in = gr.Video(label="Upload video")
            vid_out = gr.Video(label="Annotated output")
        vid_summary = gr.Markdown()
        gr.Button("Run tracking + compliance", variant="primary").click(
            run_video, inputs=vid_in, outputs=[vid_out, vid_summary])

if __name__ == "__main__":
    demo.launch()