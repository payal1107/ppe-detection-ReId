# PPE Detection and Re-ID on CCTV

Detects PPE (helmet, vest, mask, gloves) on CCTV footage, tracks each worker, and recognises a worker again when they return (Re-ID).
Internship project by Payal Gupta.

## Folders

```
ppe_detection/ppe_demo_app.py   PPE detection app (Gradio)
reid/reid_demo_v5.py            Re-ID demo
tools/                          helper scripts (cut a clip, find people in a video)
models/                         trained model files
demo_videos/reid/               Re-ID demo videos
notebooks/                      Kaggle notebooks (training and evaluation)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run all commands from this folder.

## Run the PPE app

```bash
python ppe_detection/ppe_demo_app.py
```

Open http://127.0.0.1:7860

- **Image tab:** upload an image to see PPE detections and violations.
- **Video tab:** upload a video. Each person gets an ID and a compliance result. You can download the annotated video.

The whole video is processed. Change `TARGET_FPS` or `MAX_VIDEO_MINUTES` at the top of `ppe_demo_app.py` to make it faster or shorter.

## Run the Re-ID demo

Open `reid/reid_demo_v5.py` and set `VIDEO` (input clip) and `OUT` (output file) at the top, then run it from the repository root.

```bash
python reid/reid_demo_v5.py
```

Each person is shown as `Person N`. If someone leaves and comes back, the same number is kept. Recorded results are in `demo_videos/reid/`.

## Models

| File | Use |
| --- | --- |
| `phase3_best.pt` | PPE detector (YOLO11s, 10 classes) |
| `osnet_ain_x1_0_with_cam1.pt` | Re-ID model (OSNet-AIN) |
| `yolov8n.pt` | Person detector for the PPE app (COCO) |
| `yolov8s.pt` | Person detector for the Re-ID demo (COCO) |

## Results

- PPE detection: 91.3% violation accuracy, mAP@0.5 0.642.
- Tracking: BoT-SORT gives MOTA 66.3% (ByteTrack 58.5%).
- Re-ID: Rank-1 69.0% on Cam_12, 31.9% on an unseen camera (Cam_3).

## Limitations

- The PPE model was trained on public images, not on this site's CCTV.
- Re-ID works worse on cameras it has not seen.
- Workers in identical uniforms can get a new ID after a long gap.
