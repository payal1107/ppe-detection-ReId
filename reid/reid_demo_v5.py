import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort

# ===== Ye settings badal sakti ho =====
VIDEO       = "clip_cam12.mp4"          # RAW clip
REID_W      = "models/osnet_ain_x1_0_with_cam1.pt"
OUT         = "final_cam_output12.mp4"
SIM_THRESH  = 0.70   # match ke liye kam se kam itni similarity
MARGIN      = 0.08   # best match 2nd best se kam se kam itna aage ho
PENDING_EMB = 5      # naye person ke kitne achhe frames dekh ke decide kare
MIN_BOX_H   = 80     # isse chhote box (pixels) embedding ke liye skip
DET_MODEL   = "models/yolov8s.pt"   # n se better: group ko alag-alag log pakadta hai
MAX_ASPECT  = 0.8    # box chaudai/lambai isse zyada = shayad 2-3 log ek box me
MAX_VIEWS   = 30     # har person ke kitne alag "views" yaad rakhe
RECHECK_WINDOW_SEC = 30   # naye person ka faisla itne sec tak dobara check hoga
LATE_THRESH = 0.70      # late RE-ID ke liye alag threshold (galat merge ho to badhao, jaise 0.78)
LATE_MARGIN = 0.15      # late RE-ID tabhi jab best match 2nd se itna aage ho (confusion se bachne ke liye)
GAP_SEC     = 3         # tracker ne kisi ko itne sec ke liye khoya aur...
MAX_JUMP    = 0.12      # ...naya track purani jagah ke paas (frame width ka 12%) aaya -> same person
GAP_MIN_SIM = 0.45      # bas itna check ki appearance bilkul alag na ho
IGNORE_BOX  = [385, 84, 622, 381]
# ======================================

EMB_EVERY = 3
MIN_EMB   = 5

det = YOLO(DET_MODEL)
tracker = BotSort(reid_weights=Path(REID_W), device="cpu", half=False,
                  track_buffer=150, proximity_thresh=0.9, appearance_thresh=0.5)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 10
w, h = int(cap.get(3)), int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
PENDING_MAX = int(fps * 2)   # 2 sec se zyada wait nahi

local2global, pending, seen = {}, {}, {}
gallery, count, relink_until, views = {}, {}, {}, {}
first_seen, last_seen, created_new = {}, {}, {}
last_center, pend_start = {}, {}
raw_ids = set()
next_gid, n_relinks = 1, 0
prev_active = set()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def embed(reid, box, frame):
    f = reid.get_features(np.array([box], dtype=np.float32), frame)
    f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
    return unit(f[0])


def add_view(g, e):
    """naya view tabhi jodo jab woh purane views se thoda alag ho (alag angle/pose)"""
    vs = views.setdefault(g, [])
    if not vs or max(float(v @ e) for v in vs) < 0.92:
        vs.append(e)
        if len(vs) > MAX_VIEWS:
            vs.pop(0)


def score(g, q):
    """mean embedding + best 3 views ka average -> angle change pe bhi match"""
    sims = sorted((float(v @ q) for v in views.get(g, [])), reverse=True)[:3]
    top = sum(sims) / len(sims) if sims else -1.0
    return 0.5 * float(gallery[g] @ q) + 0.5 * top


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def is_good(box, boxes, k):
    x1, y1, x2, y2 = box
    if (y2 - y1) < MIN_BOX_H or (x2 - x1) < 25:
        return False
    if (x2 - x1) / (y2 - y1) > MAX_ASPECT:              # group box
        return False
    if x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5:   # kinare pe kata hua
        return False
    for j, b in enumerate(boxes):
        if j != k and iou(box, b) > 0.2:               # dusre se overlap
            return False
    return True


def decide(tid, embs, active_now, frame_no):
    global next_gid, n_relinks
    sims = []
    q = None
    if embs:
        q = unit(np.mean(embs, axis=0))
        for g, e in gallery.items():
            if count[g] >= MIN_EMB and g not in active_now and g not in prev_active:
                sims.append((score(g, q), g))
        sims.sort(reverse=True)
    best_s, best_g = sims[0] if sims else (-1.0, None)
    second = sims[1][0] if len(sims) > 1 else -1.0

    # 1) short-gap link: tracker ne thodi der ke liye khoya, person usi jagah ke paas wapas mila
    link_g, link_d = None, None
    if tid in pend_start:
        fs, (cx, cy) = pend_start.pop(tid)
        for og, ls in last_seen.items():
            if og in active_now or og in prev_active or og not in last_center:
                continue
            if not (0 < fs - ls <= int(fps * GAP_SEC)):
                continue
            d = ((cx - last_center[og][0]) ** 2 + (cy - last_center[og][1]) ** 2) ** 0.5
            if d > MAX_JUMP * w:
                continue
            if q is not None and og in gallery and score(og, q) < GAP_MIN_SIM:
                continue
            if link_d is None or d < link_d:
                link_g, link_d = og, d

    if link_g is not None:
        g = link_g
        n_relinks += 1
        relink_until[g] = frame_no + int(fps * 3)
        tag = f"LINK(short gap, {link_d:.0f}px)"
    elif best_g is not None and best_s >= SIM_THRESH and (best_s - second) >= MARGIN:
        g = best_g
        n_relinks += 1
        relink_until[g] = frame_no + int(fps * 3)
        tag = "RE-ID"
    else:
        g = next_gid
        next_gid += 1
        tag = "NEW"
        created_new[g] = frame_no
    print(f"frame {frame_no}: tracker ID {tid} -> {tag} Person {g} "
          f"(best {best_s:.2f}, 2nd {second:.2f}, frames used {len(embs)})")

    local2global[tid] = g
    for e in embs:
        add_view(g, e)
    if q is not None:
        if g in gallery:
            c = count[g]
            gallery[g] = unit(gallery[g] * c + q * len(embs))
            count[g] = c + len(embs)
        else:
            gallery[g], count[g] = q, len(embs)
    return g


def late_score(o, g):
    """purane person o vs naye person g: mean + sabse milte-julte views"""
    cross = sorted((float(a @ b) for a in views.get(o, []) for b in views.get(g, [])), reverse=True)[:3]
    top = sum(cross) / len(cross) if cross else -1.0
    return 0.5 * float(gallery[o] @ gallery[g]) + 0.5 * top


def late_relink(frame_no):
    """NEW person ka faisla dobara check karo: kya ye koi purana (gaya hua) person hai?"""
    global n_relinks
    for g, born in list(created_new.items()):
        if frame_no - born > int(fps * RECHECK_WINDOW_SEC):
            created_new.pop(g)
            continue
        if g not in prev_active or count.get(g, 0) < MIN_EMB:
            continue
        sims = []
        for o in gallery:
            if o == g or count[o] < MIN_EMB:
                continue
            if last_seen.get(o, 10**9) >= first_seen.get(g, -1):   # dono ek saath dikhe -> alag log
                continue
            sims.append((late_score(o, g), o))
        if not sims:
            continue
        sims.sort(reverse=True)
        best_s, o = sims[0]
        second = sims[1][0] if len(sims) > 1 else -1.0
        if best_s >= LATE_THRESH and (best_s - second) >= LATE_MARGIN:
            for t, gg in list(local2global.items()):
                if gg == g:
                    local2global[t] = o
            for e in views.get(g, []):
                add_view(o, e)
            c1, c2 = count[o], count[g]
            gallery[o] = unit(gallery[o] * c1 + gallery[g] * c2)
            count[o] = c1 + c2
            last_seen[o] = max(last_seen[o], last_seen[g])
            for d in (gallery, count, views, first_seen, last_seen, created_new, last_center):
                d.pop(g, None)
            prev_active.discard(g)
            prev_active.add(o)
            relink_until[o] = frame_no + int(fps * 3)
            n_relinks += 1
            print(f"frame {frame_no}: LATE RE-ID Person {g} -> Person {o} "
                  f"(score {best_s:.2f}, 2nd {second:.2f})")


frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = det(frame, classes=[0], conf=0.3, verbose=False)[0]
    dets = r.boxes.data.cpu().numpy() if len(r.boxes) else np.empty((0, 6))
    if len(dets) and IGNORE_BOX[2] > IGNORE_BOX[0]:
        cx = (dets[:, 0] + dets[:, 2]) / 2
        cy = (dets[:, 1] + dets[:, 3]) / 2
        bx = IGNORE_BOX
        dets = dets[~((cx >= bx[0]) & (cx <= bx[2]) & (cy >= bx[1]) & (cy <= bx[3]))]
    tracks = tracker.update(dets, frame)
    if frame_no % max(1, int(fps)) == 0:
        late_relink(frame_no)
    reid = getattr(tracker, "_reid_model", None)

    boxes = [[float(v) for v in t[:4]] for t in tracks]
    active_now = set()
    for k, t in enumerate(tracks):
        box = boxes[k]
        tid = int(t[4])
        raw_ids.add(tid)
        seen[tid] = seen.get(tid, 0) + 1

        is_new = tid not in local2global
        emb = None
        if reid is not None and (is_new or frame_no % EMB_EVERY == 0) and is_good(box, boxes, k):
            emb = embed(reid, box, frame)

        x1, y1, x2, y2 = [int(v) for v in box]
        if is_new:
            pend = pending.setdefault(tid, [])
            pend_start.setdefault(tid, (frame_no, ((x1 + x2) / 2, y2)))
            if emb is not None:
                pend.append(emb)
            if len(pend) >= PENDING_EMB or seen[tid] >= PENDING_MAX:
                decide(tid, pending.pop(tid), active_now, frame_no)
                emb = None   # already used in decision
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 2)
                cv2.putText(frame, "?", (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
                continue

        g = local2global[tid]
        active_now.add(g)
        first_seen.setdefault(g, frame_no)
        last_seen[g] = frame_no
        last_center[g] = ((x1 + x2) / 2, y2)
        if emb is not None:
            add_view(g, emb)
            if g in gallery:
                c = count[g]
                gallery[g] = unit(gallery[g] * c + emb)
                count[g] = c + 1
            else:
                gallery[g], count[g] = emb, 1

        color = tuple(int(c) for c in np.random.default_rng(g).integers(60, 255, 3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"Person {g}"
        if relink_until.get(g, -1) >= frame_no:
            label += "  RE-ID"
        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    prev_active = active_now
    info = f"Tracker IDs: {len(raw_ids)} | Persons after Re-ID: {len(set(local2global.values()))} | Re-links: {n_relinks}"
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out.write(frame)

    frame_no += 1
    if frame_no % 100 == 0:
        print(f"{frame_no}/{total} frames done")

cap.release()
out.release()
print("\n===== SUMMARY =====")
print(f"Tracker ne diye IDs      : {len(raw_ids)}")
print(f"Re-ID ke baad persons    : {len(set(local2global.values()))}")
print(f"Re-links (wapas pehchane): {n_relinks}")
print(f"Output saved             : {OUT}")
