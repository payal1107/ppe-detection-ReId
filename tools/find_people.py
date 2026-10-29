import cv2
from ultralytics import YOLO

VIDEO = "cam_12_video.mp4"            # apne video ka naam
POSTER_BOX = [385, 84, 622, 381]     # Cam_12 poster wala fake person (ignore)
STEP_SEC = 2                         # har 2 second pe check

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 10
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = int(fps * STEP_SEC)


def mmss(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def in_poster(b):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return POSTER_BOX[0] <= cx <= POSTER_BOX[2] and POSTER_BOX[1] <= cy <= POSTER_BOX[3]


hits = []
for f in range(0, total, step):
    for _ in range(step - 1):           # aage badho bina decode (fast)
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        break
    boxes = model(frame, classes=[0], conf=0.35, verbose=False)[0].boxes.xyxy.tolist()
    n = sum(1 for b in boxes if not in_poster(b))
    if n > 0:
        hits.append((f / fps, n))

print(f"Video length: {mmss(total / fps)}")
if not hits:
    print("Is video me koi person nahi mila.")
else:
    print("\nIn time ranges me log dikh rahe hain:")
    start, last, mx = hits[0][0], hits[0][0], hits[0][1]
    for t, n in hits[1:] + [(None, 0)]:
        if t is not None and t - last <= STEP_SEC * 3:
            last, mx = t, max(mx, n)
            continue
        print(f"  {mmss(start)}  se  {mmss(last + STEP_SEC)}   (max {mx} log)")
        if t is not None:
            start, last, mx = t, t, n
