# AI Dance Teacher

A real-time AI dance coach that uses **YOLOv8 pose estimation** to watch you dance, identify your moves, and give you step-by-step instructions — like having a personal dance teacher in your camera.

## What it does

- Detects your body pose frame-by-frame using YOLOv8
- Classifies dance moves in real-time (Hip-Hop, Latin, Ballet, Contemporary…)
- Displays the **current move name** on screen
- Shows **what you should do** (teaching instruction) for each move
- Tells you **what comes next** so you can plan ahead
- Tracks a live **move sequence** strip at the bottom of the screen
- Prints a full **session summary** with your complete move sequence and detected dance style when you quit

## Supported moves

| Move | Style |
|---|---|
| Hands Up | Hip-Hop |
| Arms Wide | Contemporary |
| Deep Squat | Hip-Hop |
| Plié | Ballet |
| Hip Sway Left / Right | Latin / Salsa |
| Knee Lift Left / Right | Hip-Hop |
| Torso Lean | Contemporary |
| Ready Stance | General |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Webcam (default)
python main.py

# Mirrored webcam — easier to follow along
python main.py --mirror

# Local video file
python main.py --source dance_video.mp4

# YouTube video
python main.py --source "https://www.youtube.com/watch?v=..."

# Save the annotated output
python main.py --source dance_video.mp4 --output result.mp4

# Use a more accurate (but slower) model
python main.py --model yolov8s-pose.pt
```

## Controls

| Key | Action |
|---|---|
| `Q` | Quit and print session summary |

## Session summary example

```
============================================================
  AI DANCE TEACHER — SESSION SUMMARY
============================================================
  Dance Style Detected : Latin / Salsa
  Total Moves Tracked  : 12

  Move Sequence:
     1. Ready Stance
     2. Hip Sway Left
     3. Hip Sway Right x3
     4. Knee Lift Left
     5. Hip Sway Right
     6. Arms Wide
============================================================
```
