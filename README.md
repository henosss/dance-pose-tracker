# Squat Form Coach

A mobile-friendly web app that watches your squat through your phone's camera and gives
real-time feedback on your form — depth, torso lean, and knee alignment — plus a
per-rep score. Everything runs locally in the browser using [MediaPipe Pose
Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker);
no video is ever uploaded anywhere.

## Features

- Real-time skeleton overlay on your camera feed
- Automatic rep counting via a knee-angle state machine
- Live coaching cues: "go lower", "chest up", "push your knees out"
- Per-rep score (0–100) with a running rep history
- Side View mode (depth + forward lean) and Front View mode (knee cave-in / valgus)
- Installable as a home-screen app (PWA) for a native-app feel

## Running it on your phone

Camera access requires a secure context (HTTPS, or `localhost`). The easiest options:

1. **GitHub Pages** — push this repo and enable Pages (Settings → Pages → deploy from
   `main`). Open the resulting `https://<user>.github.io/<repo>/` URL on your phone.
2. **Local network over HTTPS** — serve the folder with any static HTTPS server and
   open it from your phone on the same network.

Once open in your phone's browser, use "Add to Home Screen" (Safari) or "Install app"
(Chrome) to pin it like a regular app.

## Local development

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` on the same machine (camera works over plain HTTP
on `localhost`). To test from an actual phone during development, tunnel it over HTTPS
(e.g. with `ngrok http 8000`) since phones need a secure context.

## How to use

1. Prop your phone up about 1.5–2.5m away with your full body in frame, in good
   lighting.
2. Pick **Side View** to check squat depth and forward lean, or **Front View** to check
   if your knees cave inward.
3. Tap **Start Camera**, then **Start Analyzing**.
4. Squat — you'll get live cues and a score after each rep.

## Tech

Plain HTML/CSS/JS, no build step. Pose detection via `@mediapipe/tasks-vision`
(loaded from CDN), landmark angle math in [`js/poseUtils.js`](js/poseUtils.js), app
logic and the rep state machine in [`js/app.js`](js/app.js).

This is a form guide, not medical or professional coaching advice.
