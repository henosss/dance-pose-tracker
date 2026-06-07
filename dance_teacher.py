from collections import deque, Counter
from dance_classifier import STYLE_NAMES


class DanceTeacher:
    """Tracks detected moves over time and produces teaching feedback."""

    def __init__(self, history_seconds=3, fps=30, stability_frames=8):
        self._buf_size = int(history_seconds * fps)
        self._stability = stability_frames  # frames a move must hold before announced
        self.history = deque(maxlen=self._buf_size)
        self.sequence = []          # confirmed move sequence (name only)
        self._pending = None        # (move_name, count)
        self.current_move = None
        self.current_instruction = ""
        self.current_next_hint = ""
        self._style_votes = Counter()

    def update(self, move, score):
        name = move["name"]
        self.history.append(name)

        # Stability filter: only confirm a move after it holds for N frames
        if self._pending and self._pending[0] == name:
            count = self._pending[1] + 1
            self._pending = (name, count)
            if count == self._stability and name != self.current_move:
                self.current_move = name
                self.current_instruction = move["instruction"]
                self.current_next_hint = move["next_hint"]
                self.sequence.append(name)
                self._style_votes[move["style"]] += 1
        else:
            self._pending = (name, 1)

    @property
    def dominant_style(self):
        if not self._style_votes:
            return "General Dance"
        style_key = self._style_votes.most_common(1)[0][0]
        return STYLE_NAMES.get(style_key, style_key)

    def print_session_summary(self):
        print("\n" + "=" * 60)
        print("  AI DANCE TEACHER — SESSION SUMMARY")
        print("=" * 60)
        print(f"  Dance Style Detected : {self.dominant_style}")
        print(f"  Total Moves Tracked  : {len(self.sequence)}")
        print()
        if self.sequence:
            print("  Move Sequence:")
            # Collapse consecutive duplicates for readability
            collapsed = []
            for m in self.sequence:
                if not collapsed or collapsed[-1][0] != m:
                    collapsed.append([m, 1])
                else:
                    collapsed[-1][1] += 1
            for i, (m, cnt) in enumerate(collapsed, 1):
                repeat = f" ×{cnt}" if cnt > 1 else ""
                print(f"    {i:>2}. {m}{repeat}")
        else:
            print("  No distinct moves were detected this session.")
        print("=" * 60)
        print()
