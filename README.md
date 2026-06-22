# Deutsch Verben — German Verb Flashcards

A lightweight, install-free flashcard app for learning German verbs with
**spaced repetition** and **progress tracking**.

## Features

- **400 verbs**, curated into two decks:
  - **200 B1/B2 core verbs** — everyday German verbs you need for fluency.
  - **200 C1/C2 English-cognate verbs** — advanced verbs that share a
    recognizable root with English (e.g. *analysieren → analyse*,
    *kritisieren → criticize*), so you build vocabulary fast.
- **Spaced repetition** using the SM-2 algorithm (the same family used by
  Anki). Cards you find hard come back sooner; cards you know are pushed out
  days or weeks.
- **Progress tracking** — cards learned, cards due, and a daily study streak,
  all saved in your browser's `localStorage` (per device, no account needed).
- **Two study directions** — German → English or English → German.
- Keyboard friendly: `Space` to reveal, `1`–`4` to grade (Again / Hard / Good / Easy).

## Run it

No build step, no dependencies. Just open the app:

```bash
# from this folder
python3 -m http.server 8000
# then visit http://localhost:8000
```

Or simply open `index.html` directly in a browser.

## Files

- `index.html` — the entire app (UI + spaced-repetition engine).
- `words.js` — the 400-verb dataset (`{ de, en, level, group }`).

## How grading works

| Button | Quality | Effect |
|--------|---------|--------|
| Again  | 0 | Reset; reappears within the session |
| Hard   | 3 | Short interval, smaller ease |
| Good   | 4 | Standard SM-2 interval growth |
| Easy   | 5 | Longer interval, ease boost |

Progress is stored under the key `deutsch-verben-v1`. Use **Reset progress** in
the header to start over.
