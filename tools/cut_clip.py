import cv2

VIDEO = "cam_12_video.mp4"   # apne video ka asli naam
START = "7:46"             # kahan se shuru (MM:SS)
END   = "13:36"             # kahan tak (MM:SS)
OUT   = "clip_cam12.mp4"

def to_sec(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
print(f"Video length: {int(total_sec//60)}:{int(total_sec%60):02d}  |  fps: {fps:.1f}")

start_f = int(to_sec(START) * fps)
end_f   = int(to_sec(END) * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(end_f - start_f):
    ok, frame = cap.read()
    if not ok: break
    out.write(frame)
cap.release(); out.release()
print(f"{OUT} ready: {START} se {END} tak")