# German B2 Verb Flashcards

A self-contained flashcard app for studying 310 high-yield German verbs (B2 level and the strong/irregular verbs that underpin it). No build step, no dependencies — just open `index.html` in a browser, or serve the folder with any static file server.

## Running it

```bash
cd flashcards
python3 -m http.server 8000
# open http://localhost:8000
```

Or just double-click `index.html`.

## Features

- Flip cards: infinitive on the front; Präsens (3rd person singular), Präteritum, Perfekt, English meaning, and an example sentence with translation on the back.
- Rate each card "Again" or "Know it" to track progress (New → Learning → Known), saved in `localStorage`.
- Filter the study set (all / new / learning / known) and search by German or English.
- Shuffle, keyboard shortcuts (Space/Enter flip, ←/→ navigate, K know, A again), light/dark theme toggle.

## Data

`verbs.js` holds the verb list as a plain array — edit it directly to add, remove, or correct entries.
