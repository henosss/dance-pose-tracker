(function () {
  "use strict";

  const STORAGE_KEY = "de-b2-flashcards-progress-v1";
  const THEME_KEY = "de-b2-flashcards-theme";

  const els = {
    front: document.getElementById("frontInf"),
    backInf: document.getElementById("backInf"),
    backEn: document.getElementById("backEn"),
    backPres: document.getElementById("backPres"),
    backPret: document.getElementById("backPret"),
    backPerfekt: document.getElementById("backPerfekt"),
    backEx: document.getElementById("backEx"),
    backExEn: document.getElementById("backExEn"),
    card: document.getElementById("flashcard"),
    counter: document.getElementById("cardCounter"),
    search: document.getElementById("search"),
    modeSelect: document.getElementById("modeSelect"),
    shuffleBtn: document.getElementById("shuffleBtn"),
    resetBtn: document.getElementById("resetBtn"),
    prevBtn: document.getElementById("prevBtn"),
    nextBtn: document.getElementById("nextBtn"),
    againBtn: document.getElementById("againBtn"),
    knowBtn: document.getElementById("knowBtn"),
    themeToggle: document.getElementById("themeToggle"),
    statTotal: document.getElementById("statTotal"),
    statKnown: document.getElementById("statKnown"),
    statLearning: document.getElementById("statLearning"),
    statNew: document.getElementById("statNew"),
    progressFill: document.getElementById("progressFill"),
  };

  let progress = loadProgress();
  let deck = [];
  let index = 0;

  function loadProgress() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    } catch (e) {
      return {};
    }
  }

  function saveProgress() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }

  function statusOf(verb) {
    return progress[verb.inf] || "new";
  }

  function updateStats() {
    let known = 0, learning = 0, fresh = 0;
    for (const v of VERBS) {
      const s = statusOf(v);
      if (s === "known") known++;
      else if (s === "learning") learning++;
      else fresh++;
    }
    els.statTotal.textContent = VERBS.length;
    els.statKnown.textContent = known;
    els.statLearning.textContent = learning;
    els.statNew.textContent = fresh;
    els.progressFill.style.width = (VERBS.length ? (known / VERBS.length) * 100 : 0) + "%";
  }

  function buildDeck() {
    const mode = els.modeSelect.value;
    const q = els.search.value.trim().toLowerCase();
    deck = VERBS.filter((v) => {
      if (mode !== "all" && statusOf(v) !== mode) return false;
      if (q && !v.inf.toLowerCase().includes(q) && !v.en.toLowerCase().includes(q)) return false;
      return true;
    });
    index = 0;
    render();
  }

  function currentVerb() {
    return deck[index];
  }

  function render() {
    els.card.classList.remove("flipped");
    if (!deck.length) {
      els.front.textContent = "No cards";
      els.counter.textContent = "0 / 0";
      els.backInf.textContent = "—";
      els.backEn.textContent = "Try a different filter or search.";
      els.backPres.textContent = "—";
      els.backPret.textContent = "—";
      els.backPerfekt.textContent = "—";
      els.backEx.textContent = "";
      els.backExEn.textContent = "";
      return;
    }
    const v = currentVerb();
    els.front.textContent = v.inf;
    els.backInf.textContent = v.inf;
    els.backEn.textContent = v.en;
    els.backPres.textContent = v.pres;
    els.backPret.textContent = v.pret;
    els.backPerfekt.textContent = v.perfekt;
    els.backEx.textContent = v.ex;
    els.backExEn.textContent = v.exEn;
    els.counter.textContent = `${index + 1} / ${deck.length}`;
  }

  function flip() {
    if (!deck.length) return;
    els.card.classList.toggle("flipped");
  }

  function go(delta) {
    if (!deck.length) return;
    index = (index + delta + deck.length) % deck.length;
    render();
  }

  function rate(status) {
    if (!deck.length) return;
    const v = currentVerb();
    progress[v.inf] = status;
    saveProgress();
    updateStats();
    go(1);
  }

  function shuffleDeck() {
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    index = 0;
    render();
  }

  function resetProgress() {
    if (!confirm("Reset all study progress? This can't be undone.")) return;
    progress = {};
    saveProgress();
    updateStats();
    buildDeck();
  }

  function applyTheme(theme) {
    if (theme) {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const currentlyDark = current ? current === "dark" : prefersDark;
    const next = currentlyDark ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem(THEME_KEY, next);
  }

  els.card.addEventListener("click", flip);
  els.card.addEventListener("keydown", (e) => {
    if (e.code === "Space" || e.code === "Enter") {
      e.preventDefault();
      flip();
    }
  });
  els.prevBtn.addEventListener("click", () => go(-1));
  els.nextBtn.addEventListener("click", () => go(1));
  els.knowBtn.addEventListener("click", () => rate("known"));
  els.againBtn.addEventListener("click", () => rate("learning"));
  els.shuffleBtn.addEventListener("click", shuffleDeck);
  els.resetBtn.addEventListener("click", resetProgress);
  els.search.addEventListener("input", buildDeck);
  els.modeSelect.addEventListener("change", buildDeck);
  els.themeToggle.addEventListener("click", toggleTheme);

  document.addEventListener("keydown", (e) => {
    if (document.activeElement === els.search) return;
    if (e.code === "ArrowRight") go(1);
    else if (e.code === "ArrowLeft") go(-1);
    else if (e.key === "k" || e.key === "K") rate("known");
    else if (e.key === "a" || e.key === "A") rate("learning");
    else if (e.code === "Space" || e.code === "Enter") {
      if (document.activeElement !== els.card) {
        e.preventDefault();
        flip();
      }
    }
  });

  const savedTheme = localStorage.getItem(THEME_KEY);
  if (savedTheme) applyTheme(savedTheme);

  updateStats();
  buildDeck();
})();
