import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { extractMetrics } from "./poseUtils.js";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startOverlay = document.getElementById("startOverlay");
const startBtn = document.getElementById("startBtn");
const toggleBtn = document.getElementById("toggleBtn");
const switchCamBtn = document.getElementById("switchCamBtn");
const feedbackBanner = document.getElementById("feedbackBanner");
const repCountEl = document.getElementById("repCount");
const lastScoreEl = document.getElementById("lastScore");
const kneeAngleEl = document.getElementById("kneeAngleReadout");
const repListEl = document.getElementById("repList");
const sideViewBtn = document.getElementById("sideViewBtn");
const frontViewBtn = document.getElementById("frontViewBtn");
const helpBtn = document.getElementById("helpBtn");
const helpModal = document.getElementById("helpModal");
const closeHelpBtn = document.getElementById("closeHelpBtn");

let poseLandmarker = null;
let running = false;
let analyzing = false;
let currentFacing = "environment";
let currentStream = null;
let viewMode = "side"; // 'side' | 'front'
let lastVideoTime = -1;

// ---- Rep state machine ----
const UP_THRESHOLD = 160; // knee angle considered "standing"
const DOWN_THRESHOLD = 150; // knee angle considered "in a squat" (catches shallow attempts too)
const MIN_DEPTH_FOR_REP = 150; // must reach at least this angle to count as a rep

let repState = "up"; // 'up' | 'down'
let repCount = 0;
let curRep = null; // { minKnee, maxLean, valgusFlag }
let smoothedKnee = null;

const SMOOTHING = 0.35; // exponential smoothing factor for angle jitter

function resetCurRep() {
  curRep = { minKnee: 180, maxLean: 0, valgusFlagged: false };
}

function gradeRep(rep) {
  let score = 100;
  let notes = [];

  if (rep.minKnee > 130) {
    score -= 40;
    notes.push("Too shallow");
  } else if (rep.minKnee > 110) {
    score -= 20;
    notes.push("Shallow — go lower");
  } else if (rep.minKnee > 100) {
    score -= 8;
    notes.push("Near parallel");
  } else {
    notes.push("Great depth");
  }

  if (rep.maxLean > 45) {
    score -= 25;
    notes.push("Too much forward lean");
  } else if (rep.maxLean > 30) {
    score -= 10;
    notes.push("Keep chest up");
  }

  if (rep.valgusFlagged) {
    score -= 20;
    notes.push("Knees caved in");
  }

  score = Math.max(0, Math.min(100, Math.round(score)));

  let grade = "poor";
  if (score >= 85) grade = "excellent";
  else if (score >= 70) grade = "good";
  else if (score >= 50) grade = "needswork";

  return { score, grade, notes };
}

function addRepToHistory(rep, result) {
  const emptyHint = repListEl.querySelector(".empty-hint");
  if (emptyHint) emptyHint.remove();

  const li = document.createElement("li");
  li.className = "rep-item";
  li.innerHTML = `
    <span class="rep-num">Rep ${repCount}</span>
    <span class="rep-notes">${result.notes.join(" · ")}</span>
    <span class="rep-score grade-${result.grade}">${result.score}</span>
  `;
  repListEl.prepend(li);
}

function setFeedback(text, state) {
  feedbackBanner.textContent = text;
  feedbackBanner.classList.remove("hidden", "state-good", "state-warn", "state-bad", "state-neutral");
  feedbackBanner.classList.add(`state-${state}`);
}

function processMetrics(metrics) {
  if (!metrics || metrics.kneeAngle == null) {
    kneeAngleEl.textContent = "--°";
    return;
  }

  smoothedKnee =
    smoothedKnee == null
      ? metrics.kneeAngle
      : smoothedKnee + SMOOTHING * (metrics.kneeAngle - smoothedKnee);

  kneeAngleEl.textContent = `${Math.round(smoothedKnee)}°`;

  const valgusActive =
    viewMode === "front" &&
    metrics.valgusRatio != null &&
    metrics.valgusRatio < 0.8;

  // State transitions
  if (repState === "up" && smoothedKnee < DOWN_THRESHOLD) {
    repState = "down";
    resetCurRep();
  } else if (repState === "down" && smoothedKnee > UP_THRESHOLD) {
    repState = "up";
    if (curRep && curRep.minKnee <= MIN_DEPTH_FOR_REP) {
      repCount += 1;
      repCountEl.textContent = String(repCount);
      const result = gradeRep(curRep);
      lastScoreEl.textContent = String(result.score);
      addRepToHistory(curRep, result);
      setFeedback(
        result.score >= 70 ? `Nice rep! ${result.notes[0]}` : result.notes.join(" · "),
        result.score >= 85 ? "good" : result.score >= 50 ? "warn" : "bad"
      );
    }
    curRep = null;
  }

  if (repState === "down" && curRep) {
    curRep.minKnee = Math.min(curRep.minKnee, smoothedKnee);
    if (metrics.hipLean != null) curRep.maxLean = Math.max(curRep.maxLean, metrics.hipLean);
    if (valgusActive) curRep.valgusFlagged = true;

    if (valgusActive) {
      setFeedback("Push your knees out", "bad");
    } else if (metrics.hipLean != null && metrics.hipLean > 45) {
      setFeedback("Chest up — don't lean forward", "warn");
    } else if (smoothedKnee <= 100) {
      setFeedback("Great depth — drive back up", "good");
    } else {
      setFeedback("Keep going down...", "neutral");
    }
  } else if (repState === "up") {
    setFeedback(repCount > 0 ? "Ready for next rep" : "Squat when ready", "neutral");
  }
}

function drawResults(result) {
  ctx.save();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const drawer = new DrawingUtils(ctx);
  if (result.landmarks) {
    for (const landmarks of result.landmarks) {
      drawer.drawLandmarks(landmarks, { radius: 3, color: "#38bdf8" });
      drawer.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: "#22c55e",
        lineWidth: 3,
      });
    }
  }
  ctx.restore();
}

function resizeCanvasToVideo() {
  overlay.width = video.videoWidth || overlay.clientWidth;
  overlay.height = video.videoHeight || overlay.clientHeight;
}

function renderLoop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const nowMs = performance.now();
    const result = poseLandmarker.detectForVideo(video, nowMs);
    drawResults(result);

    if (analyzing) {
      if (result.landmarks && result.landmarks.length > 0) {
        processMetrics(extractMetrics(result.landmarks[0]));
      } else {
        setFeedback("Can't see your full body — step back", "warn");
      }
    }
  }
  requestAnimationFrame(renderLoop);
}

async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function startCamera(facing) {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facing },
      width: { ideal: 720 },
      height: { ideal: 1280 },
    },
  };
  currentStream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = currentStream;
  video.classList.toggle("mirrored", facing === "user");
  overlay.classList.toggle("mirrored", facing === "user");
  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resizeCanvasToVideo();
      resolve();
    };
  });
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Loading model...";
  try {
    await Promise.all([initPoseLandmarker(), startCamera(currentFacing)]);
    startOverlay.classList.add("hidden");
    toggleBtn.disabled = false;
    switchCamBtn.disabled = false;
    running = true;
    renderLoop();
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    startBtn.textContent = "Start Camera";
    alert(
      "Couldn't start the camera. Make sure you're on HTTPS (or localhost) and grant camera permission."
    );
  }
});

toggleBtn.addEventListener("click", () => {
  analyzing = !analyzing;
  toggleBtn.textContent = analyzing ? "Stop Analyzing" : "Start Analyzing";
  toggleBtn.classList.toggle("recording", analyzing);
  if (analyzing) {
    feedbackBanner.classList.remove("hidden");
    repState = "up";
    curRep = null;
    smoothedKnee = null;
    setFeedback("Squat when ready", "neutral");
  } else {
    feedbackBanner.classList.add("hidden");
  }
});

switchCamBtn.addEventListener("click", async () => {
  currentFacing = currentFacing === "environment" ? "user" : "environment";
  switchCamBtn.disabled = true;
  try {
    await startCamera(currentFacing);
  } catch (err) {
    console.error(err);
    alert("Couldn't switch camera.");
  } finally {
    switchCamBtn.disabled = false;
  }
});

function setViewMode(mode) {
  viewMode = mode;
  sideViewBtn.classList.toggle("active", mode === "side");
  frontViewBtn.classList.toggle("active", mode === "front");
}
sideViewBtn.addEventListener("click", () => setViewMode("side"));
frontViewBtn.addEventListener("click", () => setViewMode("front"));

helpBtn.addEventListener("click", () => helpModal.classList.remove("hidden"));
closeHelpBtn.addEventListener("click", () => helpModal.classList.add("hidden"));
helpModal.addEventListener("click", (e) => {
  if (e.target === helpModal) helpModal.classList.add("hidden");
});

window.addEventListener("resize", resizeCanvasToVideo);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  });
}
