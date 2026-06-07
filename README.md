# Dance Pose Tracker

A computer vision application that uses **YOLOv9** to detect and classify dance moves from video, frame-by-frame.

## Features

- **YOLOv9 pose detection** — 17-keypoint COCO skeleton on every frame
- **Dance move classifier** — rule-based classifier with temporal smoothing:
  - Arms Raised
  - Jump / Airborne
  - Squat / Low
  - Spin / Twist
  - Side Stretch
  - Standing
- **YouTube support** — download any dance video via URL with `yt-dlp`
- **Annotated output video** — skeleton overlay, per-person bounding box, HUD with move label and FPS
- **Move statistics** — frame-level breakdown printed at the end

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Process a YouTube video
python dance_pose_tracker.py --url "https://www.youtube.com/watch?v=<id>" --output dance_out.mp4

# 2b. Process a local video file
python dance_pose_tracker.py --video path/to/dance.mp4 --output dance_out.mp4

# 2c. Run the synthetic demo (no video needed)
python demo.py
```

The YOLOv9 model weights (`yolov9c-pose.pt`) are **automatically downloaded** the first time you run the tracker.

## CLI reference

```
usage: dance_pose_tracker.py [-h] (--url URL | --video VIDEO)
                              [--output OUTPUT] [--model MODEL]
                              [--conf CONF] [--device DEVICE]
                              [--show] [--max-frames N]
                              [--download-dir DIR]

options:
  --url URL          YouTube URL to download and process
  --video VIDEO      Path to a local video file
  --output, -o       Output annotated video path (default: output_dance.mp4)
  --model            YOLOv9 pose weights (default: yolov9c-pose.pt)
  --conf             Detection confidence threshold (default: 0.4)
  --device           Inference device: '' auto, 'cpu', '0' GPU (default: '')
  --show             Display live preview window while processing
  --max-frames N     Limit to first N frames, 0 = all (default: 0)
  --download-dir     Directory for downloaded YouTube videos (default: downloads)
```

## How it works

```
YouTube URL ──► yt-dlp ──► local MP4
                                │
                        OpenCV frame loop
                                │
                         YOLOv9 predict()
                        (yolov9c-pose.pt)
                                │
                    17-keypoint skeleton per person
                                │
                  DanceMoveClassifier (rule-based)
                  + temporal smoothing (8-frame window)
                                │
                   draw skeleton + HUD + bbox label
                                │
                        output annotated MP4
                                │
                    print move statistics to terminal
```

## Output example

```
── Dance Move Statistics ────────────────────────────────────
  Arms Raised            ████████████████████████  48.2%  (289 frames)
  Standing               ██████████████            28.5%  (171 frames)
  Squat / Low            ████████                  16.3%  ( 98 frames)
  Spin / Twist           ██                         5.0%  ( 30 frames)
  Jump / Airborne        █                          2.0%  ( 12 frames)
─────────────────────────────────────────────────────────────
```

## Model

The default model is `yolov9c-pose.pt` — YOLOv9-C fine-tuned for human pose estimation. You can swap in any Ultralytics-compatible YOLOv9 pose model:

```bash
python dance_pose_tracker.py --video dance.mp4 --model yolov9e-pose.pt
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list
