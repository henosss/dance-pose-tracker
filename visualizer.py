import cv2
import numpy as np

# Skeleton connections for drawing
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COLOURS = {
    "skeleton": (0, 200, 255),
    "joint": (255, 255, 255),
    "box_bg": (0, 0, 0),
    "move_text": (0, 255, 180),
    "instruction_text": (220, 220, 220),
    "hint_text": (255, 200, 80),
    "seq_text": (180, 180, 255),
    "style_text": (255, 120, 200),
}


def draw_skeleton(frame, kp, conf_thresh=0.3):
    h, w = frame.shape[:2]
    pts = []
    for i in range(17):
        x, y, c = kp[i]
        pts.append((int(x), int(y), c))

    for a, b in SKELETON:
        xa, ya, ca = pts[a]
        xb, yb, cb = pts[b]
        if ca > conf_thresh and cb > conf_thresh:
            cv2.line(frame, (xa, ya), (xb, yb), COLOURS["skeleton"], 2, cv2.LINE_AA)

    for x, y, c in pts:
        if c > conf_thresh:
            cv2.circle(frame, (x, y), 4, COLOURS["joint"], -1, cv2.LINE_AA)


def _text_box(frame, lines, x, y, line_height=28, padding=10, alpha=0.55):
    """Draw a semi-transparent black box behind multiple text lines."""
    max_w = max(cv2.getTextSize(l[0], cv2.FONT_HERSHEY_SIMPLEX, l[1], 1)[0][0] for l in lines)
    box_h = len(lines) * line_height + padding * 2
    box_w = max_w + padding * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - padding, y - padding),
                  (x + box_w, y + box_h), COLOURS["box_bg"], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hud(frame, teacher, sequence_window=6):
    h, w = frame.shape[:2]

    # ── Top-left panel ──
    move = teacher.current_move or "Waiting…"
    style = teacher.dominant_style
    instruction = teacher.current_instruction or "Stand in frame and start dancing!"
    hint = teacher.current_next_hint or ""

    lines = [
        (f"MOVE: {move}", 0.75, COLOURS["move_text"]),
        (f"STYLE: {style}", 0.55, COLOURS["style_text"]),
        (f"{instruction}", 0.50, COLOURS["instruction_text"]),
    ]
    if hint:
        lines.append((f"Next: {hint}", 0.48, COLOURS["hint_text"]))

    y_cursor = 30
    for text, scale, colour in lines:
        # word-wrap long lines
        words = text.split()
        line, wrapped = [], []
        for word in words:
            test = " ".join(line + [word])
            tw, _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
            if tw > w - 20:
                wrapped.append((" ".join(line), scale, colour))
                line = [word]
            else:
                line.append(word)
        if line:
            wrapped.append((" ".join(line), scale, colour))
        for t, s, c in wrapped:
            cv2.putText(frame, t, (12, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, t, (12, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
            y_cursor += int(scale * 38)

    # ── Bottom: sequence strip ──
    recent = teacher.sequence[-sequence_window:]
    if recent:
        seq_text = "  ▶  ".join(recent)
        label = f"Sequence: {seq_text}"
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        bx, by = 10, h - 18
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 4, by - th - 6), (bx + tw + 4, by + 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, label, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOURS["seq_text"], 1, cv2.LINE_AA)
