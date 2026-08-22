// MediaPipe Pose landmark indices we care about.
export const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

const VISIBILITY_MIN = 0.5;

function visible(pt) {
  return pt && (pt.visibility === undefined || pt.visibility >= VISIBILITY_MIN);
}

// Angle at vertex `b` formed by points a-b-c, in degrees (0-180).
export function angleDeg(a, b, c) {
  const abx = a.x - b.x, aby = a.y - b.y;
  const cbx = c.x - b.x, cby = c.y - b.y;
  const magAB = Math.hypot(abx, aby);
  const magCB = Math.hypot(cbx, cby);
  if (magAB === 0 || magCB === 0) return null;
  const cos = (abx * cbx + aby * cby) / (magAB * magCB);
  return (Math.acos(Math.min(1, Math.max(-1, cos))) * 180) / Math.PI;
}

// Deviation of the shoulder->hip line from vertical, in degrees. 0 = upright torso.
export function leanAngle(shoulder, hip) {
  const dx = Math.abs(hip.x - shoulder.x);
  const dy = Math.abs(hip.y - shoulder.y);
  if (dx === 0 && dy === 0) return null;
  return (Math.atan2(dx, dy) * 180) / Math.PI;
}

// Picks the more visible side (or averages both) and returns a consistent
// set of derived metrics for form analysis. `landmarks` are normalized
// (0-1) image coordinates from PoseLandmarker.
export function extractMetrics(landmarks) {
  const L = landmarks;
  const leftSide = [L[LM.LEFT_SHOULDER], L[LM.LEFT_HIP], L[LM.LEFT_KNEE], L[LM.LEFT_ANKLE]];
  const rightSide = [L[LM.RIGHT_SHOULDER], L[LM.RIGHT_HIP], L[LM.RIGHT_KNEE], L[LM.RIGHT_ANKLE]];
  const leftOk = leftSide.every(visible);
  const rightOk = rightSide.every(visible);

  if (!leftOk && !rightOk) return null;

  let kneeAngle, hipLean;
  if (leftOk && rightOk) {
    const kL = angleDeg(L[LM.LEFT_HIP], L[LM.LEFT_KNEE], L[LM.LEFT_ANKLE]);
    const kR = angleDeg(L[LM.RIGHT_HIP], L[LM.RIGHT_KNEE], L[LM.RIGHT_ANKLE]);
    kneeAngle = kL !== null && kR !== null ? (kL + kR) / 2 : kL ?? kR;
    const lL = leanAngle(L[LM.LEFT_SHOULDER], L[LM.LEFT_HIP]);
    const lR = leanAngle(L[LM.RIGHT_SHOULDER], L[LM.RIGHT_HIP]);
    hipLean = lL !== null && lR !== null ? (lL + lR) / 2 : lL ?? lR;
  } else {
    const [sh, hip, knee, ankle] = leftOk ? leftSide : rightSide;
    kneeAngle = angleDeg(hip, knee, ankle);
    hipLean = leanAngle(sh, hip);
  }

  // Knee valgus: ratio of knee separation to ankle separation. Only
  // meaningful when both sides of the body are visible (front-ish view).
  let valgusRatio = null;
  if (leftOk && rightOk) {
    const kneeWidth = Math.hypot(
      L[LM.LEFT_KNEE].x - L[LM.RIGHT_KNEE].x,
      L[LM.LEFT_KNEE].y - L[LM.RIGHT_KNEE].y
    );
    const ankleWidth = Math.hypot(
      L[LM.LEFT_ANKLE].x - L[LM.RIGHT_ANKLE].x,
      L[LM.LEFT_ANKLE].y - L[LM.RIGHT_ANKLE].y
    );
    if (ankleWidth > 0.001) valgusRatio = kneeWidth / ankleWidth;
  }

  return { kneeAngle, hipLean, valgusRatio, bothSidesVisible: leftOk && rightOk };
}
