const state = { view: "summary", sort: "overall", direction: "desc", query: "", type: "all" };
const languages = ["DE", "EN", "ES", "FR", "ID", "IT", "NL", "PT", "VI", "AR", "HI", "JP", "KO", "RU", "TH", "ZH", "ZH-T"];
const summaryColumns = [["overall", "Overall"], ["digital", "Digital"], ["photo", "Photo"], ["latin", "Latin"], ["non_latin", "Non-Latin"], ["private", "Private"]];

function value(score) { return Number.isFinite(score) ? score.toFixed(1) : "—"; }
function metadata(model) { return MODEL_METADATA[model.name] || { label: model.name, links: [] }; }

function linkIcon(label) {
  if (label === "GitHub") return `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.2a6.8 6.8 0 0 0-2.15 13.25c.34.06.46-.15.46-.33v-1.2c-1.87.4-2.26-.8-2.26-.8-.3-.78-.75-.99-.75-.99-.61-.42.05-.41.05-.41.67.05 1.03.7 1.03.7.6 1.02 1.56.73 1.94.56.06-.43.23-.73.43-.9-1.49-.17-3.05-.74-3.05-3.31 0-.73.26-1.32.69-1.79-.07-.17-.3-.85.07-1.77 0 0 .56-.18 1.84.68A6.37 6.37 0 0 1 8 3.96a6.4 6.4 0 0 1 1.67.22c1.27-.86 1.83-.68 1.83-.68.38.92.14 1.6.07 1.77.43.47.69 1.06.69 1.79 0 2.58-1.57 3.14-3.06 3.3.24.21.45.6.45 1.21v1.8c0 .18.12.4.46.33A6.8 6.8 0 0 0 8 1.2Z"/></svg>`;
  return `<span class="hf-icon" aria-hidden="true">🤗</span>`;
}

function modelLinks(model) {
  const links = metadata(model).links || [];
  if (!links.length) return "";
  return `<span class="model-links">${links.map((link) => `<a class="model-link" href="${link.url}" target="_blank" rel="noreferrer" aria-label="${link.label}" title="${link.label}">${linkIcon(link.label)}</a>`).join("")}</span>`;
}

function columnsForView() { return state.view === "languages" ? languages.map((language) => [language, language]) : summaryColumns; }
function scoreFor(model, key) { return languages.includes(key) ? model.languages[key] : model[key]; }
function sortIndicator(key) {
  if (state.sort !== key) return `<span class="sort-indicator" aria-hidden="true">↕</span>`;
  return `<span class="sort-indicator active" aria-hidden="true">${state.direction === "desc" ? "↓" : "↑"}</span>`;
}

function renderHeader() {
  const headings = columnsForView().map(([key, label]) => `<th><button class="sort-button" data-column="${key}">${label}${sortIndicator(key)}</button></th>`).join("");
  document.querySelector("#leaderboard-head").innerHTML = `<tr><th>Rank</th><th>Model</th><th>Type</th>${headings}</tr>`;
}

function render(data) {
  const models = data.models
    .filter((m) => state.type === "all" || m.type === state.type)
    .filter((m) => `${m.name} ${m.source_name || ""}`.toLowerCase().includes(state.query))
    .sort((a, b) => {
      const difference = scoreFor(a, state.sort) - scoreFor(b, state.sort);
      return state.direction === "desc" ? -difference : difference;
    });
  renderHeader();
  document.querySelector("#leaderboard-body").innerHTML = models.map((model, index) => {
    const scores = columnsForView().map(([key]) => `<td class="score${key === "overall" ? " featured" : ""}">${value(scoreFor(model, key))}</td>`).join("");
    return `<tr class="model-row"><td class="rank">${index + 1}</td><td><span class="model-name">${metadata(model).label}</span>${modelLinks(model)}</td><td><span class="type type-${model.type.toLowerCase().replaceAll(" ", "-")}">${model.type}</span></td>${scores}</tr>`;
  }).join("");
  document.querySelectorAll("[data-column]").forEach((button) => button.addEventListener("click", () => {
    const column = button.dataset.column;
    state.direction = state.sort === column && state.direction === "desc" ? "asc" : "desc";
    state.sort = column;
    render(data);
  }));
}

fetch("leaderboard.json")
  .then((response) => response.json())
  .then((data) => {
    document.querySelector("#updated").textContent = `Last updated · ${data.benchmark.updated}`;
    render(data);
    document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
      state.view = button.dataset.view;
      state.sort = state.view === "languages" ? "EN" : "overall";
      state.direction = "desc";
      document.querySelectorAll("[data-view]").forEach((item) => item.classList.toggle("active", item === button));
      render(data);
    }));
    document.querySelector("#search").addEventListener("input", (event) => { state.query = event.target.value.toLowerCase().trim(); render(data); });
    document.querySelector("#type-filter").addEventListener("change", (event) => { state.type = event.target.value; render(data); });
  })
  .catch(() => { document.querySelector("#leaderboard-body").innerHTML = '<tr><td colspan="20" class="error">Could not load leaderboard data.</td></tr>'; });
