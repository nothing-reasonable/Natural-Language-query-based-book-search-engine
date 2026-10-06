"use strict";

const content = document.querySelector("#content");
const announcer = document.querySelector("#announcer");
const headerSearch = document.querySelector("#header-search");
const headerQuery = document.querySelector("#header-query");
const cardTemplate = document.querySelector("#book-card-template");

let ready = false;
let engineFailure = "";
let facets = null;
let activeRequest = null;
let requestNumber = 0;
let engineStatus = {};

document.addEventListener("click", event => {
  const link = event.target.closest("a[data-link], a[data-book-link]");
  if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const href = link.getAttribute("href");
  if (!href || !href.startsWith("/")) return;
  event.preventDefault();
  navigate(href);
});

headerSearch.addEventListener("submit", event => {
  event.preventDefault();
  const query = headerQuery.value.trim();
  if (query) navigate(searchUrl({q: query, page: 1}));
});
window.addEventListener("popstate", renderRoute);
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  const panel = document.querySelector("#filters.open");
  if (panel) {
    panel.classList.remove("open");
    const toggle = document.querySelector("#filter-toggle");
    toggle?.setAttribute("aria-expanded", "false");
    toggle?.focus();
  }
});

boot();

async function boot() {
  renderRoute();
  await waitForEngine();
  if (!facets) {
    try { facets = await api("/api/facets"); } catch { facets = emptyFacets(); }
  }
  renderRoute();
}

async function waitForEngine() {
  while (!ready) {
    try {
      const status = await api("/api/status");
      engineStatus = status;
      ready = !!status.ready;
      announce(status.message || status.stage);
      if (status.state === "failed") {
        engineFailure = status.message || "অনুসন্ধান চালু করা যায়নি।";
        return;
      }
    } catch { announce("সার্ভারের সঙ্গে সংযোগের অপেক্ষা চলছে।"); }
    if (!ready) await delay(1200);
  }
}

function renderRoute() {
  if (activeRequest) activeRequest.abort();
  const path = location.pathname;
  const params = new URLSearchParams(location.search);
  headerSearch.hidden = path === "/" && !params.get("q") && !params.has("browse");
  headerQuery.value = params.get("q") || "";
  document.title = "বইখোঁজ — বাংলা বই আবিষ্কার";

  if (path.startsWith("/books/")) {
    renderBook(decodeURIComponent(path.slice(7)));
  } else if (params.get("q")) {
    renderSearch(params);
  } else if (params.has("browse")) {
    renderBrowse(params);
  } else {
    renderHome();
  }
}

function renderHome() {
  content.innerHTML = `
    <section class="hero">
      <p class="kicker">আপনার পরের বইটি খুঁজে নিন</p>
      <h1>শব্দ নয়, ভাব দিয়ে<br>বাংলা বই খুঁজুন</h1>
      <p>লেখক, বিষয়, প্রকাশকাল—অথবা মনের কথাটি স্বাভাবিক ভাষায় লিখুন।</p>
      <form class="hero-search" id="hero-search" role="search">
        <label class="sr-only" for="hero-query">বই খুঁজুন</label>
        <input id="hero-query" type="search" maxlength="300" autofocus
               placeholder="যেমন: মুক্তিযুদ্ধ নিয়ে লেখা স্মৃতিকথা" autocomplete="off">
        <button type="submit">বই খুঁজুন</button>
      </form>
      <div class="examples" id="examples" aria-label="উদাহরণ অনুসন্ধান"></div>
    </section>
    <section class="section shell" aria-labelledby="subject-heading">
      <div class="section-head"><div><p class="eyebrow">বিষয় ধরে</p><h2 id="subject-heading">আপনার আগ্রহের জগৎ</h2></div></div>
      <div class="subject-cloud" id="subject-cloud"><span>ক্যাটালগ প্রস্তুত হচ্ছে…</span></div>
    </section>
    <section class="section shell" aria-labelledby="selection-heading">
      <div class="section-head"><div><p class="eyebrow">ক্যাটালগ থেকে</p><h2 id="selection-heading">কিছু বই দেখে নিন</h2></div><a href="/?browse=1" data-link>সব বই →</a></div>
      <div class="book-grid" id="home-books">${skeletons(3)}</div>
    </section>`;

  const examples = [
    "মুক্তিযুদ্ধ নিয়ে লেখা স্মৃতিকথা",
    "হুমায়ূন আহমেদের উপন্যাস",
    "১৯৯০ থেকে ২০০০ সালে প্রকাশিত বই",
  ];
  const box = document.querySelector("#examples");
  examples.forEach(query => {
    const button = node("button", "", query);
    button.type = "button";
    button.addEventListener("click", () => navigate(searchUrl({q: query, page: 1})));
    box.append(button);
  });
  document.querySelector("#hero-search").addEventListener("submit", event => {
    event.preventDefault();
    const query = document.querySelector("#hero-query").value.trim();
    if (query) navigate(searchUrl({q: query, page: 1}));
  });
  if (engineFailure) {
    document.querySelector("#subject-cloud").textContent = engineFailure;
    document.querySelector("#home-books").replaceChildren(stateBox(engineFailure, true));
  } else if (ready) loadHomeContent();
}

async function loadHomeContent() {
  const cloud = document.querySelector("#subject-cloud");
  const grid = document.querySelector("#home-books");
  if (!cloud || !grid) return;
  cloud.innerHTML = "";
  (facets?.subjects || []).slice(0, 12).forEach(item => {
    const link = node("a", "subject-pill", `${item.value} · ${bn(item.count)}`);
    link.href = searchUrl({q: item.value, subject: item.value, page: 1});
    link.dataset.link = "";
    cloud.append(link);
  });
  try {
    const data = await api("/api/catalog?page=1&page_size=24&order=year_desc");
    if (!document.querySelector("#home-books")) return;
    grid.innerHTML = "";
    const useful = data.items.filter(book => book.description).slice(0, 6);
    (useful.length ? useful : data.items.slice(0, 6)).forEach(book => grid.append(bookCard(book)));
  } catch (error) { grid.replaceChildren(stateBox(error.message, true)); }
}

async function renderSearch(params) {
  const query = params.get("q") || "";
  document.title = `${query} — বইখোঁজ`;
  const view = params.get("view") === "list" ? "list" : "grid";
  const showLlmReranker = engineStatus.reranker_configured === "lmstudio";
  const llmRerankerAvailable = engineStatus.reranker_backend === "lmstudio";
  const rerank = rerankRequested(params);
  content.innerHTML = `
    <section class="search-page shell">
      <div class="search-heading">
        <div><p class="eyebrow">অনুসন্ধান</p><h1 id="search-title"></h1><p class="result-note" id="result-note">ফলাফল প্রস্তুত হচ্ছে…</p></div>
        <div class="toolbar">
          <button class="tool-button filter-toggle" id="filter-toggle" aria-expanded="false">ফিল্টার</button>
          ${showLlmReranker ? `<button class="tool-button rerank-toggle" id="rerank-toggle"
                  aria-pressed="${rerank}" ${llmRerankerAvailable ? "" : "disabled"}
                  title="${llmRerankerAvailable ? "এই অনুসন্ধানে LM Studio পুনঃক্রম চালু বা বন্ধ করুন" : "LM Studio পুনঃক্রম এখন উপলব্ধ নয়"}">
                  LM Studio: ${rerank ? "চালু" : "বন্ধ"}</button>` : ""}
          <button class="tool-button" data-view="grid" aria-pressed="${view === "grid"}">গ্রিড</button>
          <button class="tool-button" data-view="list" aria-pressed="${view === "list"}">তালিকা</button>
        </div>
      </div>
      <div class="filter-chips" id="filter-chips"></div>
      <div class="search-layout">
        <aside class="filters" id="filters" aria-label="ফলাফল ছাঁকুন"></aside>
        <div><div class="book-grid ${view}" id="search-results">${skeletons(6)}</div><nav class="pagination" id="pagination" aria-label="ফলাফলের পাতা"></nav></div>
      </div>
    </section>`;
  document.querySelector("#search-title").textContent = `“${query}”`;
  bindViewButtons(view);
  bindRerankToggle(params, rerank);
  buildFilters(params);
  buildChips(params);

  if (!ready) {
    document.querySelector("#result-note").textContent = engineFailure || "ইঞ্জিন প্রস্তুত হওয়ার অপেক্ষা চলছে…";
    if (engineFailure) document.querySelector("#search-results").replaceChildren(stateBox(engineFailure, true));
    return;
  }

  const number = ++requestNumber;
  activeRequest = new AbortController();
  try {
    const payload = {
      query,
      page: positive(params.get("page"), 1),
      page_size: 12,
      rerank,
      filters: filtersFromParams(params),
    };
    const data = await api("/api/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
      signal: activeRequest.signal,
    });
    if (number !== requestNumber) return;
    renderSearchResults(data, view);
  } catch (error) {
    if (error.name === "AbortError" || number !== requestNumber) return;
    document.querySelector("#search-results")?.replaceChildren(stateBox(error.message, true));
    announce(error.message);
  }
}

function renderSearchResults(data, view) {
  const results = document.querySelector("#search-results");
  const note = document.querySelector("#result-note");
  if (!results || !note) return;
  results.className = `book-grid ${view}`;
  results.innerHTML = "";
  note.textContent = `${bn(data.ranked_count)}টি সাজানো ফলাফল · পাতা ${bn(data.page)}`;
  if (data.rerank?.fallback) note.textContent += " · ফিউশন ক্রম ব্যবহৃত";
  if (!data.hits.length) {
    results.append(stateBox("এই শর্তে কোনো বই পাওয়া যায়নি। ফিল্টার বদলে আবার চেষ্টা করুন।"));
  } else {
    data.hits.forEach(hit => results.append(bookCard(hit, {rank: hit.rank, search: true})));
  }
  renderPagination(data.page, data.pages);
  announce(`${data.hits.length}টি বই দেখানো হচ্ছে।`);
}

function buildFilters(params) {
  const aside = document.querySelector("#filters");
  if (!aside) return;
  aside.innerHTML = `<button class="tool-button filter-toggle" id="filter-close">বন্ধ করুন</button><h2>ফলাফল ছাঁকুন</h2>`;
  aside.append(
    filterInput("লেখক", "author", params.get("author") || "", "authors-list"),
    filterInput("প্রকাশক", "publisher", params.get("publisher") || "", "publishers-list"),
    filterSelect("বিষয়", "subject", params.get("subject") || "", facets?.subjects || []),
    filterSelect("ধরন", "genre", params.get("genre") || "", facets?.genres || []),
  );
  const years = node("div", "filter-field");
  years.append(node("label", "", "প্রকাশকাল"));
  const row = node("div", "year-row");
  row.append(yearInput("year_from", params.get("year_from") || "", "শুরু"),
             yearInput("year_to", params.get("year_to") || "", "শেষ"));
  years.append(row);
  aside.append(years);
  const clear = node("button", "clear-filters", "সব ফিল্টার মুছুন");
  clear.type = "button";
  clear.addEventListener("click", () => navigate(searchUrl({
    q: params.get("q"), view: params.get("view"), rerank: params.get("rerank"),
  })));
  aside.append(clear, dataList("authors-list", facets?.authors || []), dataList("publishers-list", facets?.publishers || []));

  aside.querySelectorAll("input,select").forEach(control => control.addEventListener("change", () => {
    const next = new URLSearchParams(location.search);
    if (control.value.trim()) next.set(control.name, control.value.trim()); else next.delete(control.name);
    next.set("page", "1");
    navigate(`/search?${next}`);
  }));
  const toggle = document.querySelector("#filter-toggle");
  toggle?.addEventListener("click", () => {
    aside.classList.add("open");
    toggle.setAttribute("aria-expanded", "true");
    aside.querySelector("input,select")?.focus();
  });
  aside.querySelector("#filter-close")?.addEventListener("click", () => {
    aside.classList.remove("open");
    toggle?.setAttribute("aria-expanded", "false");
    toggle?.focus();
  });
}

function buildChips(params) {
  const box = document.querySelector("#filter-chips");
  if (!box) return;
  const labels = {author: "লেখক", publisher: "প্রকাশক", subject: "বিষয়", genre: "ধরন", year_from: "শুরু", year_to: "শেষ"};
  Object.keys(labels).forEach(key => {
    const value = params.get(key);
    if (!value) return;
    const chip = node("span", "chip", `${labels[key]}: ${value}`);
    const remove = node("button", "", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `${labels[key]} ফিল্টার মুছুন`);
    remove.addEventListener("click", () => {
      const next = new URLSearchParams(location.search);
      next.delete(key); next.set("page", "1"); navigate(`/search?${next}`);
    });
    chip.append(remove); box.append(chip);
  });
}

async function renderBrowse(params) {
  const page = positive(params.get("page"), 1);
  content.innerHTML = `<section class="search-page shell"><div class="search-heading"><div><p class="eyebrow">সম্পূর্ণ ক্যাটালগ</p><h1>সব বই</h1><p class="result-note" id="result-note">ক্যাটালগ প্রস্তুত হচ্ছে…</p></div></div><div class="book-grid" id="browse-results">${skeletons(6)}</div><nav class="pagination" id="pagination"></nav></section>`;
  if (!ready) {
    if (engineFailure) document.querySelector("#browse-results").replaceChildren(stateBox(engineFailure, true));
    return;
  }
  try {
    const data = await api(`/api/catalog?page=${page}&page_size=12&order=title_asc`);
    const grid = document.querySelector("#browse-results");
    if (!grid) return;
    grid.innerHTML = "";
    data.items.forEach(book => grid.append(bookCard(book)));
    document.querySelector("#result-note").textContent = `ক্যাটালগে ${bn(data.total)}টি বই`;
    renderPagination(data.page, data.pages, true);
  } catch (error) { document.querySelector("#browse-results")?.replaceChildren(stateBox(error.message, true)); }
}

async function renderBook(bookId) {
  content.innerHTML = `<section class="shell detail"><div class="skeleton"></div><div><div class="skeleton"></div></div></section>`;
  if (!ready) {
    if (engineFailure) content.replaceChildren(stateBox(engineFailure, true));
    return;
  }
  activeRequest = new AbortController();
  try {
    const book = await api(`/api/books/${encodeURIComponent(bookId)}`, {signal: activeRequest.signal});
    document.title = `${book.title} — বইখোঁজ`;
    content.innerHTML = `<article class="shell detail"><div class="detail-cover" id="detail-cover"></div><div class="detail-copy"><button class="back-link link-button" id="back-button">← ফিরে যান</button><p class="eyebrow">বইয়ের বিস্তারিত</p><h1 id="detail-title"></h1><p class="detail-author" id="detail-author"></p><p class="detail-meta" id="detail-meta"></p><a id="source-link" class="detail-link" target="_blank" rel="noopener noreferrer" hidden>মূল ক্যাটালগে দেখুন ↗</a><section class="detail-section" id="description-section"><h2>বইয়ের বিবরণ</h2><p id="detail-description"></p></section><section class="detail-section" id="contents-section"><h2>সূচিপত্র</h2><p id="detail-contents"></p></section><section class="detail-section" id="tags-section"><h2>বিষয় ও ধরন</h2><p class="provenance-note">নিচের শ্রেণিগুলো ক্যাটালগের লেখা থেকে অনুমিত—এগুলো প্রকাশকের দেওয়া তথ্য নয়।</p><div class="tag-row" id="detail-tags"></div></section></div></article>`;
    document.querySelector("#detail-title").textContent = book.title || "শিরোনামহীন";
    document.querySelector("#detail-author").textContent = book.author || "লেখক অজানা";
    document.querySelector("#detail-meta").textContent = [book.publisher, book.publish_year].filter(Boolean).join(" · ");
    document.querySelector("#detail-description").textContent = book.description || "বিবরণ পাওয়া যায়নি।";
    document.querySelector("#detail-contents").textContent = book.table_of_contents || "সূচিপত্র পাওয়া যায়নি।";
    document.querySelector("#detail-cover").append(cover(book));
    document.querySelector("#back-button").addEventListener("click", () => history.back());
    if (book.source_url) {
      const sourceLink = document.querySelector("#source-link");
      sourceLink.href = book.source_url;
      sourceLink.hidden = false;
    }
    const tagBox = document.querySelector("#detail-tags");
    [...(book.inferred?.subjects || []), ...(book.inferred?.genres || [])].forEach(value => tagBox.append(node("span", "tag", value)));
    if (!tagBox.children.length) document.querySelector("#tags-section").hidden = true;
    announce(`${book.title} বইয়ের বিস্তারিত দেখানো হচ্ছে।`);
  } catch (error) {
    if (error.name !== "AbortError") content.replaceChildren(stateBox(error.message, true));
  }
}

function bookCard(book, options = {}) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);
  card.querySelectorAll("[data-book-link]").forEach(link => {
    link.href = `/books/${encodeURIComponent(book.book_id)}`;
    link.dataset.link = "";
  });
  card.querySelector(".cover").replaceWith(cover(book));
  card.querySelector(".book-title").textContent = book.title || "শিরোনামহীন";
  card.querySelector(".book-author").textContent = book.author || "লেখক অজানা";
  card.querySelector(".book-meta").textContent = [book.publisher, book.publish_year].filter(Boolean).join(" · ");
  card.querySelector(".book-description").textContent = book.description || "এই বইয়ের বিবরণ পাওয়া যায়নি।";
  if (options.rank) card.querySelector(".card-rank").textContent = `ফলাফল ${bn(options.rank)}`;
  const excerpt = card.querySelector(".match-excerpt");
  if (options.search && book.match_excerpt) {
    excerpt.hidden = false;
    const source = book.match_source === "table_of_contents" ? "সূচিপত্রে" : "বইয়ের বিবরণে";
    excerpt.textContent = `${source} মিলেছে: “${book.match_excerpt}”`;
  }
  const tags = [...(book.inferred?.subjects || []), ...(book.inferred?.genres || [])].slice(0, 4);
  tags.forEach(value => card.querySelector(".tag-row").append(node("span", "tag", value)));
  return card;
}

function cover(book) {
  const fallback = node("div", "cover placeholder");
  fallback.append(node("span", "cover-letter", (book.title || "ব").trim().charAt(0)));
  if (!book.cover_url) return fallback;
  const image = node("img", "cover");
  image.src = book.cover_url;
  image.alt = `${book.title || "বই"}-এর প্রচ্ছদ`;
  image.loading = "lazy";
  image.addEventListener("error", () => image.replaceWith(fallback), {once: true});
  return image;
}

function renderPagination(page, pages, browse = false) {
  const nav = document.querySelector("#pagination");
  if (!nav || pages <= 1) return;
  const previous = node("button", "", "← আগের পাতা");
  previous.disabled = page <= 1;
  const label = node("span", "", `${bn(page)} / ${bn(pages)}`);
  const next = node("button", "", "পরের পাতা →");
  next.disabled = page >= pages;
  const go = target => {
    const params = new URLSearchParams(location.search);
    params.set("page", String(target));
    navigate(`${browse ? "/" : "/search"}?${params}`);
    window.scrollTo({top: 0, behavior: "smooth"});
  };
  previous.addEventListener("click", () => go(page - 1));
  next.addEventListener("click", () => go(page + 1));
  nav.append(previous, label, next);
}

function bindViewButtons(current) {
  document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => {
    const params = new URLSearchParams(location.search);
    params.set("view", button.dataset.view);
    navigate(`/search?${params}`);
  }));
}

function rerankRequested(params) {
  const requested = params.get("rerank");
  if (requested === "0") return false;
  if (requested === "1") return true;
  return engineStatus.reranker_enabled !== false;
}

function bindRerankToggle(params, enabled) {
  document.querySelector("#rerank-toggle")?.addEventListener("click", () => {
    const next = new URLSearchParams(params);
    next.set("rerank", enabled ? "0" : "1");
    next.set("page", "1");
    navigate(`/search?${next}`);
  });
}

function filtersFromParams(params) {
  const out = {};
  ["author", "publisher", "subject", "genre"].forEach(key => {
    if (params.get(key)) out[key] = params.get(key);
  });
  ["year_from", "year_to"].forEach(key => {
    if (params.get(key)) out[key] = Number(params.get(key));
  });
  return out;
}

function filterInput(label, name, value, list) {
  const wrap = node("div", "filter-field");
  const lab = node("label", "", label); lab.htmlFor = `filter-${name}`;
  const input = node("input"); input.id = `filter-${name}`; input.name = name; input.value = value; input.setAttribute("list", list);
  wrap.append(lab, input); return wrap;
}

function filterSelect(label, name, value, choices) {
  const wrap = node("div", "filter-field");
  const lab = node("label", "", label); lab.htmlFor = `filter-${name}`;
  const select = node("select"); select.id = `filter-${name}`; select.name = name;
  select.append(new Option("সব", ""));
  choices.forEach(item => select.append(new Option(`${item.value} (${bn(item.count)})`, item.value)));
  select.value = value; wrap.append(lab, select); return wrap;
}

function yearInput(name, value, placeholder) {
  const input = node("input"); input.type = "number"; input.name = name; input.value = value;
  input.placeholder = placeholder; input.min = "1000"; input.max = "2200"; input.setAttribute("aria-label", placeholder);
  return input;
}

function dataList(id, choices) {
  const list = node("datalist"); list.id = id;
  choices.slice(0, 500).forEach(item => list.append(new Option(item.value, item.value)));
  return list;
}

function stateBox(message, error = false) {
  const box = node("div", `state-box${error ? " error" : ""}`);
  box.append(node("h2", "", error ? "কিছু একটা ঠিক হয়নি" : "কোনো বই পাওয়া যায়নি"), node("p", "", message));
  return box;
}

function skeletons(count) { return Array.from({length: count}, () => '<div class="skeleton" aria-hidden="true"></div>').join(""); }
function emptyFacets() { return {authors: [], publishers: [], subjects: [], genres: []}; }
function positive(value, fallback) { const number = Number(value); return Number.isInteger(number) && number > 0 ? number : fallback; }
function bn(value) { return String(value ?? "").replace(/\d/g, digit => "০১২৩৪৫৬৭৮৯"[digit]); }
function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function announce(message) { announcer.textContent = ""; requestAnimationFrame(() => { announcer.textContent = message || ""; }); }
function node(tag, className = "", text = "") { const element = document.createElement(tag); if (className) element.className = className; if (text !== "") element.textContent = text; return element; }

function searchUrl(values) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => { if (value !== undefined && value !== null && value !== "") params.set(key, value); });
  return `/search?${params}`;
}

function navigate(href) {
  history.pushState({}, "", href);
  renderRoute();
  content.focus({preventScroll: true});
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const error = new Error(data.error || "অনুরোধটি সম্পন্ন করা যায়নি।");
    error.status = response.status;
    throw error;
  }
  return data;
}
