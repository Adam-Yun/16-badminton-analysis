# Shuttle Tracker

Shuttlecock detection and tracking pipeline for badminton match videos.
Uses **TrackNetV2** (a VGG-encoder / U-Net-decoder heatmap regression model) combined with a **Kalman filter** for temporally smooth tracking at 30 fps.

---

## 1. Installation

```bash
cd shuttle-tracker
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU (CUDA) is strongly advised for training; inference works on CPU.

---

## 2. Data Preparation

### 2a. Extract frames from match videos

Place your `.mp4` / `.avi` files in `data/raw_videos/`, then run:

```bash
python pipeline/extract_frames.py data/raw_videos/match1.mp4 data/frames/match1/
```

This creates `data/frames/match1/frame_00000.jpg … frame_NNNNN.jpg` and a `metadata.json`.

### 2b. Annotations CSV

Create `data/annotations/labels.csv` with one row per frame:

| column       | type  | description                              |
|--------------|-------|------------------------------------------|
| `frame_path` | str   | Absolute or relative path to the frame   |
| `x`          | float | Normalised horizontal position (0 – 1)   |
| `y`          | float | Normalised vertical position (0 – 1)     |
| `visible`    | int   | 1 = shuttle visible, 0 = occluded/absent |

When `visible=0`, the `x` and `y` values are ignored; set them to 0.

---

## 3. Training

```bash
python pipeline/train.py \
    --data  data/annotations/labels.csv \
    --epochs 100 \
    --batch  8 \
    --lr     1e-4 \
    --n_frames 3
```

| flag          | default | description                              |
|---------------|---------|------------------------------------------|
| `--data`      | —       | Path to annotations CSV (required)       |
| `--epochs`    | 100     | Maximum training epochs                  |
| `--batch`     | 8       | Batch size                               |
| `--lr`        | 1e-4    | Initial learning rate (cosine annealing) |
| `--n_frames`  | 3       | Consecutive frames per sample            |
| `--resume`    | —       | Path to checkpoint to resume from        |

The best checkpoint is saved to `models/weights/best.pt` and the loss curve to `models/weights/loss_curve.png`.
Training stops early if validation loss does not improve for 15 consecutive epochs.

---

## 4. Inference

```bash
python pipeline/infer.py \
    --video   data/raw_videos/match2.mp4 \
    --weights models/weights/best.pt \
    --output  data/match2_tracked.mp4 \
    --n_frames 3 \
    --conf_threshold 0.5
```

| flag               | default | description                                |
|--------------------|---------|---------------------------------------------|
| `--video`          | —       | Input video file (required)                 |
| `--weights`        | —       | Model checkpoint (required)                 |
| `--output`         | —       | Annotated output video path (required)      |
| `--n_frames`       | 3       | Must match the value used during training   |
| `--conf_threshold` | 0.5     | Minimum heatmap peak confidence (0 – 1)     |

---

## 5. Expected Output

The annotated video contains:

- **Red circle** at the Kalman-smoothed shuttlecock position every frame.
- **Fading yellow-to-red trail** showing the last 20 trajectory points; older points are more transparent.
- **Confidence score** in the top-left corner.

When the model confidence is below `--conf_threshold`, the Kalman filter predicts forward using the constant-velocity motion model without a measurement update, keeping the trail smooth through brief occlusions.

---

## 6. Project Structure

```
shuttle-tracker/
├── data/
│   ├── raw_videos/          # input match videos
│   ├── frames/              # extracted frames (frame_XXXXX.jpg)
│   ├── annotations/         # labels.csv
│   └── dataset/             # train/val split (optional)
├── models/
│   ├── tracknet.py          # TrackNetV2 architecture
│   └── weights/             # best.pt, loss_curve.png
├── pipeline/
│   ├── extract_frames.py
│   ├── augment.py
│   ├── dataset.py
│   ├── train.py
│   ├── tracker.py
│   └── infer.py
├── utils/
│   ├── visualize.py
│   └── metrics.py
├── requirements.txt
└── README.md
```
