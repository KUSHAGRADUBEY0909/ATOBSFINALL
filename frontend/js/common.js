// Shared helpers: API calls, session, header, toasts. Loaded on every page.
const ATOBS = (window.ATOBS = window.ATOBS || {});
ATOBS.context = ATOBS.context || {}; // pages put route_id / bus_registration here for the emergency form

ATOBS.api = async function (path, opts = {}) {
  const o = { credentials: "same-origin", ...opts };
  if (o.json !== undefined) {
    o.method = o.method || "POST";
    o.headers = { "Content-Type": "application/json", ...(o.headers || {}) };
    o.body = JSON.stringify(o.json);
    delete o.json;
  }
  const res = await fetch(path, o);
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok) {
    let msg = (data && data.detail) || res.statusText;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
};

ATOBS.esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
ATOBS.qs = (k) => new URLSearchParams(location.search).get(k);
ATOBS.toast = function (msg, isErr = false) {
  const t = document.createElement("div");
  t.className = "toast" + (isErr ? " err" : "");
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3800);
};
ATOBS.mins = (m) => (m == null ? "–" : m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`);
ATOBS.crowdClass = (p) => (p >= 85 ? "bad" : p >= 65 ? "warn" : "ok");
ATOBS.meter = (p) => `<div class="meter ${ATOBS.crowdClass(p) === "ok" ? "" : ATOBS.crowdClass(p)}"><i style="width:${Math.min(100, p || 0)}%"></i></div>`;
ATOBS.ago = function (ts) {
  if (!ts) return "never";
  const d = (Date.now() - ATOBS.parse(ts).getTime()) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return ATOBS.fmt(ts);
};
// DB timestamps are UTC ("YYYY-MM-DD HH:MM:SS"); show them in the viewer's local time
ATOBS.parse = (ts) => new Date(ts.replace(" ", "T") + (/[zZ+]/.test(ts.slice(10)) ? "" : "Z"));
ATOBS.fmt = (ts) => (ts ? ATOBS.parse(ts).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "–");

ATOBS.user = null;
ATOBS.ready = (async function () {
  try { ATOBS.user = (await ATOBS.api("/api/auth/me")).user; } catch (_) { ATOBS.user = null; }
  renderHeader();
  const need = document.body.dataset.role; // e.g. "operator,admin"
  if (need) {
    const roles = need.split(",");
    if (!ATOBS.user) { location.href = `/login.html?next=${encodeURIComponent(location.pathname + location.search)}`; return; }
    if (!roles.includes(ATOBS.user.role)) {
      document.querySelector("main").innerHTML = `<div class="card empty"><h2>Staff only</h2><p>This page is for ${roles.join(" / ")} accounts.</p><a class="btn" href="/">Back to home</a></div>`;
      throw new Error("forbidden");
    }
  }
  return ATOBS.user;
})();

function renderHeader() {
  const el = document.getElementById("top");
  if (!el) return;
  const u = ATOBS.user;
  const page = document.body.dataset.page;
  const links = [["home", "/", "Routes"], ["timings", "/timings.html", "Bus timings"], ["sim", "/simulation.html", "Camera simulation"], ["demo", "/demo.html", "Live demo"]];
  if (u) links.push(["alerts", "/alerts.html", "My alerts"]);
  else links.push(["alerts", "/alerts.html", "Track alert"]);
  if (u && ["operator", "admin"].includes(u.role)) links.push(["counter", "/counter.html", "Passenger counter"]);
  if (u && ["operator", "admin"].includes(u.role)) links.push(["admin", "/admin.html", 'Control room <span class="badge" id="navBadge" hidden></span>']);
  el.className = "topbar";
  el.innerHTML = `<div class="wrap">
    <a class="logo" href="/"><span class="mark">A</span>ATOBS</a>
    <nav class="nav">${links.map(([k, h, t]) => `<a href="${h}" class="${k === page ? "active" : ""}">${t}</a>`).join("")}</nav>
    <div class="userbox">${u
      ? `<span>${ATOBS.esc(u.name)}</span><span class="role">${u.role}</span><button class="btn sm ghost" style="color:#fff;border-color:rgba(255,255,255,.3)" id="logoutBtn">Log out</button>`
      : `<a class="btn sm" href="/login.html">Log in</a><a class="btn sm primary" style="background:var(--accent);border-color:var(--accent);color:var(--brand-2)" href="/register.html">Sign up</a>`}</div>
  </div>`;
  const lo = document.getElementById("logoutBtn");
  if (lo) lo.onclick = async () => { await ATOBS.api("/api/auth/logout", { method: "POST" }); location.href = "/"; };
  if (u && ["operator", "admin"].includes(u.role)) pollBadge();
}

async function pollBadge() {
  const b = document.getElementById("navBadge");
  if (!b) return;
  try {
    const s = await ATOBS.api("/api/admin/summary");
    b.hidden = !s.open_alerts;
    b.textContent = s.open_alerts;
  } catch (_) { /* ignore */ }
  setTimeout(pollBadge, 15000);
}

// Stop-name autocomplete used by the planner and counter
ATOBS.autocomplete = function (input, onPick) {
  const wrap = input.parentElement;
  wrap.classList.add("ac");
  const menu = document.createElement("div");
  menu.className = "ac-menu";
  menu.hidden = true;
  wrap.appendChild(menu);
  let timer, items = [], idx = -1;
  const pick = (it) => { input.value = it.short_name; menu.hidden = true; onPick && onPick(it); };
  input.addEventListener("input", () => {
    clearTimeout(timer);
    input.dataset.picked = "";
    timer = setTimeout(async () => {
      const q = input.value.trim();
      if (q.length < 2) { menu.hidden = true; return; }
      items = (await ATOBS.api(`/api/stops/search?q=${encodeURIComponent(q)}`)).items;
      idx = -1;
      menu.innerHTML = items.map((it, i) => `<div data-i="${i}"><span>${ATOBS.esc(it.short_name)}</span><small>${it.routes} routes</small></div>`).join("") || '<div class="muted">No stops found</div>';
      menu.hidden = false;
    }, 180);
  });
  input.addEventListener("keydown", (e) => {
    if (menu.hidden || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      idx = (idx + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      [...menu.children].forEach((c, i) => c.classList.toggle("on", i === idx));
      e.preventDefault();
    } else if (e.key === "Enter" && idx >= 0) { pick(items[idx]); e.preventDefault(); }
  });
  menu.addEventListener("mousedown", (e) => {
    const d = e.target.closest("[data-i]");
    if (d) pick(items[+d.dataset.i]);
  });
  input.addEventListener("blur", () => setTimeout(() => (menu.hidden = true), 150));
};

// Every page gets the emergency widget and the seat-map component
(function loadShared() {
  for (const src of ["/js/emergency.js", "/js/seatmap.js"]) {
    const s = document.createElement("script");
    s.src = src;
    s.async = false;
    document.head.appendChild(s);
  }
})();
ATOBS.seatsReady = new Promise((res) => {
  const t = setInterval(() => { if (ATOBS.seatMap) { clearInterval(t); res(); } }, 20);
});
ATOBS.crowdLabel = (p) => (p < 40 ? "Seats available" : p < 65 ? "Moderate" : p < 85 ? "Crowded" : "Very crowded");
ATOBS.occClass = (p) => (p >= 85 ? "occ-bad" : p >= 65 ? "occ-warn" : "occ-ok");

// Place-name autocomplete (clean names after the data clean-up) for the trip planner
ATOBS.placeAutocomplete = function (input, onPick) {
  const wrap = input.parentElement;
  wrap.classList.add("ac");
  const menu = document.createElement("div");
  menu.className = "ac-menu";
  menu.hidden = true;
  wrap.appendChild(menu);
  let timer, items = [], idx = -1;
  const pick = (it) => { input.value = it.place; menu.hidden = true; onPick && onPick(it); };
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      if (q.length < 2) { menu.hidden = true; return; }
      items = (await ATOBS.api(`/api/places/search?q=${encodeURIComponent(q)}`)).items;
      idx = -1;
      menu.innerHTML = items.map((it, i) => `<div data-i="${i}"><span>${it.terminal ? "🚏 " : ""}${ATOBS.esc(it.place)}</span><small>${it.routes} routes</small></div>`).join("") || '<div class="muted">No matching place</div>';
      menu.hidden = false;
    }, 180);
  });
  input.addEventListener("keydown", (e) => {
    if (menu.hidden || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      idx = (idx + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      [...menu.children].forEach((c, i) => c.classList.toggle("on", i === idx));
      e.preventDefault();
    } else if (e.key === "Enter" && idx >= 0) { pick(items[idx]); e.preventDefault(); }
  });
  menu.addEventListener("mousedown", (e) => { const d = e.target.closest("[data-i]"); if (d) pick(items[+d.dataset.i]); });
  input.addEventListener("blur", () => setTimeout(() => (menu.hidden = true), 150));
};
