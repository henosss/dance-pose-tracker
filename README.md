# dance-pose-tracker

A high-performance distance-running gait and biomechanics analyzer. It runs YOLO
pose estimation (`yolo26n-pose.pt`, falling back to `yolo11n-pose.pt`)
frame-by-frame over a running video and extracts efficiency metrics:

1. **Knee flexion & extension** — Hip-Knee-Ankle angle via the Law of Cosines.
2. **Vertical oscillation** — hip-centroid vertical displacement on a rolling-variance baseline.
3. **Pelvic & torso stability** — angular offset between shoulder and hip midlines.
4. **Left vs. right asymmetry** — running log of peak joint extensions to flag compensation.

Active joint angles and metrics are overlaid on the runner's skeleton, and a
biomechanical efficiency breakdown is printed to the console on completion.

## Usage

```bash
pip install ultralytics opencv-python numpy
python dance_pose_tracker.py --source run.mp4 --output annotated.mp4
```
