import numpy as np
from pose_detector import KP, PoseDetector as PD

# ---------------------------------------------------------------------------
# Each move is a dict with:
#   name        – display name
#   style       – broad dance style used for genre detection
#   check(kp)   – returns confidence 0-1
#   instruction – what to say to the student
#   next_hint   – what typically comes next
# ---------------------------------------------------------------------------

def _arm_height_ratio(kp):
    """How high are the wrists relative to shoulders (positive = above)."""
    scores = []
    for s, w in [("left_shoulder", "left_wrist"), ("right_shoulder", "right_wrist")]:
        si, wi = KP[s], KP[w]
        if PD.visible(kp, si) and PD.visible(kp, wi):
            # y decreases upward in image coords
            scores.append(PD.pt(kp, si)[1] - PD.pt(kp, wi)[1])
    return np.mean(scores) if scores else 0.0


def _knee_bend_angle(kp):
    """Average knee angle (180 = straight, <120 = deep squat)."""
    angles = []
    for hip, knee, ankle in [
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ]:
        hi, ki, ai = KP[hip], KP[knee], KP[ankle]
        if PD.visible(kp, hi) and PD.visible(kp, ki) and PD.visible(kp, ai):
            angles.append(PD.angle(PD.pt(kp, hi), PD.pt(kp, ki), PD.pt(kp, ai)))
    return np.mean(angles) if angles else 180.0


def _hip_shoulder_offset(kp):
    """Horizontal offset of hip midpoint from shoulder midpoint (normalised by shoulder width)."""
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    if all(PD.visible(kp, i) for i in [ls, rs, lh, rh]):
        sh_mid = (PD.pt(kp, ls)[0] + PD.pt(kp, rs)[0]) / 2
        hi_mid = (PD.pt(kp, lh)[0] + PD.pt(kp, rh)[0]) / 2
        sh_width = abs(PD.pt(kp, ls)[0] - PD.pt(kp, rs)[0]) + 1e-6
        return (hi_mid - sh_mid) / sh_width
    return 0.0


def _torso_lean(kp):
    """How much the torso is leaning forward (shoulder y vs hip y delta, normalised)."""
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    if all(PD.visible(kp, i) for i in [ls, rs, lh, rh]):
        sh_y = (PD.pt(kp, ls)[1] + PD.pt(kp, rs)[1]) / 2
        hi_y = (PD.pt(kp, lh)[1] + PD.pt(kp, rh)[1]) / 2
        torso_h = abs(hi_y - sh_y) + 1e-6
        sh_x = (PD.pt(kp, ls)[0] + PD.pt(kp, rs)[0]) / 2
        hi_x = (PD.pt(kp, lh)[0] + PD.pt(kp, rh)[0]) / 2
        return abs(sh_x - hi_x) / torso_h
    return 0.0


def _arm_spread(kp):
    """Wrist-to-wrist horizontal distance normalised by shoulder width."""
    lw, rw = KP["left_wrist"], KP["right_wrist"]
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    if all(PD.visible(kp, i) for i in [lw, rw, ls, rs]):
        spread = abs(PD.pt(kp, lw)[0] - PD.pt(kp, rw)[0])
        sh_width = abs(PD.pt(kp, ls)[0] - PD.pt(kp, rs)[0]) + 1e-6
        return spread / sh_width
    return 0.0


def _one_knee_up(kp):
    """Returns which knee is raised (left/right/none)."""
    for side in ["left", "right"]:
        hip_i = KP[f"{side}_hip"]
        knee_i = KP[f"{side}_knee"]
        ankle_i = KP[f"{side}_ankle"]
        if PD.visible(kp, hip_i) and PD.visible(kp, knee_i) and PD.visible(kp, ankle_i):
            hip_y = PD.pt(kp, hip_i)[1]
            knee_y = PD.pt(kp, knee_i)[1]
            ankle_y = PD.pt(kp, ankle_i)[1]
            # In image coords y grows downward; knee above hip means knee_y < hip_y
            if knee_y < hip_y - 0.05 * abs(ankle_y - hip_y + 1):
                return side
    return "none"


MOVES = [
    {
        "name": "Hands Up",
        "style": "hip-hop",
        "instruction": "Raise both hands above your head — keep arms straight and shoulders relaxed.",
        "next_hint": "Try a Hip Sway next to add groove!",
        "check": lambda kp: min(1.0, max(0.0, _arm_height_ratio(kp) / 80)),
    },
    {
        "name": "Arms Wide",
        "style": "contemporary",
        "instruction": "Spread your arms out to the sides like wings — feel the width through your fingertips.",
        "next_hint": "Now try flowing into a Torso Lean for drama.",
        "check": lambda kp: min(1.0, max(0.0, (_arm_spread(kp) - 1.2) / 1.0)),
    },
    {
        "name": "Deep Squat",
        "style": "hip-hop",
        "instruction": "Bend your knees deep — keep your back straight and feet flat on the floor.",
        "next_hint": "Explode upward into a Jump or hold the Squat Pulse.",
        "check": lambda kp: min(1.0, max(0.0, (130 - _knee_bend_angle(kp)) / 50)),
    },
    {
        "name": "Plié",
        "style": "ballet",
        "instruction": "Bend your knees gently with feet turned out — keep your torso tall.",
        "next_hint": "Rise onto your toes into a Relevé.",
        "check": lambda kp: (
            0.7 if 130 < _knee_bend_angle(kp) < 160 and _arm_spread(kp) > 0.8 else 0.0
        ),
    },
    {
        "name": "Hip Sway Left",
        "style": "latin",
        "instruction": "Shift your weight to the left — let your hip pop out naturally.",
        "next_hint": "Swing back to Hip Sway Right to start a Latin rhythm.",
        "check": lambda kp: min(1.0, max(0.0, -_hip_shoulder_offset(kp) / 0.3)),
    },
    {
        "name": "Hip Sway Right",
        "style": "latin",
        "instruction": "Shift your weight to the right — keep shoulders level as the hip pops.",
        "next_hint": "Come back to Hip Sway Left and build the groove.",
        "check": lambda kp: min(1.0, max(0.0, _hip_shoulder_offset(kp) / 0.3)),
    },
    {
        "name": "Knee Lift Left",
        "style": "hip-hop",
        "instruction": "Lift your left knee to hip height — keep the supporting leg strong.",
        "next_hint": "Step it down into a Side Step Right.",
        "check": lambda kp: 0.85 if _one_knee_up(kp) == "left" else 0.0,
    },
    {
        "name": "Knee Lift Right",
        "style": "hip-hop",
        "instruction": "Lift your right knee to hip height — engage your core to balance.",
        "next_hint": "Step it down into a Side Step Left.",
        "check": lambda kp: 0.85 if _one_knee_up(kp) == "right" else 0.0,
    },
    {
        "name": "Torso Lean",
        "style": "contemporary",
        "instruction": "Lean your upper body forward — let your arms hang loose and breathe.",
        "next_hint": "Unwind slowly back to centre for a graceful recovery.",
        "check": lambda kp: min(1.0, max(0.0, (_torso_lean(kp) - 0.15) / 0.25)),
    },
    {
        "name": "Ready Stance",
        "style": "general",
        "instruction": "Stand tall with feet shoulder-width apart — breathe and feel the music.",
        "next_hint": "When you're ready, try raising your Arms Wide to get started!",
        "check": lambda kp: max(
            0.0,
            0.5 - abs(_arm_height_ratio(kp)) / 80
                - abs(_hip_shoulder_offset(kp)) / 0.3
                - max(0, (130 - _knee_bend_angle(kp)) / 80),
        ),
    },
]

STYLE_NAMES = {
    "hip-hop": "Hip-Hop",
    "latin": "Latin / Salsa",
    "ballet": "Ballet / Contemporary",
    "contemporary": "Contemporary",
    "general": "General Dance",
}


def classify(kp):
    """Return the best-matching move dict plus its confidence score."""
    best_move = MOVES[-1]  # default: Ready Stance
    best_score = 0.0
    for move in MOVES:
        score = move["check"](kp)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move, best_score
