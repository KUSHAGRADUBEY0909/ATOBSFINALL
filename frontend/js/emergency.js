// Emergency Alert widget: fixed button at the bottom-left of every page + report panel.
(function () {
  if (document.getElementById("sosFab")) return;
  const A = window.ATOBS;
  const ICONS = { medical: "🚑", harassment: "🛡️", accident: "💥", fire: "🔥", theft: "👜", breakdown: "🔧", overcrowding: "👥", other: "❗" };
  let categories = [];
  let chosen = null;
  let coords = null;

  const fab = document.createElement("button");
  fab.id = "sosFab";
  fab.className = "sos-fab";
  fab.type = "button";
  fab.setAttribute("aria-haspopup", "dialog");
  fab.innerHTML = '<span class="ring">!</span>EMERGENCY';
  document.body.appendChild(fab);

  const ov = document.createElement("div");
  ov.className = "sos-overlay";
  ov.id = "sosOverlay";
  ov.setAttribute("role", "dialog");
  ov.setAttribute("aria-modal", "true");
  ov.setAttribute("aria-labelledby", "sosTitle");
  ov.innerHTML = `<div class="sos-panel">
    <div class="sos-head"><span style="font-size:1.3rem">🚨</span><h2 id="sosTitle">Report an emergency</h2><button type="button" id="sosClose" aria-label="Close">✕</button></div>
    <div class="sos-body" id="sosBody"></div></div>`;
  document.body.appendChild(ov);
  const body = ov.querySelector("#sosBody");

  function formHTML() {
    const u = A.user;
    const ctx = A.context || {};
    return `
      <div class="sos-call">
        <a href="tel:112">📞 112 Emergency</a><a href="tel:1091">Women 1091</a><a href="tel:108">Ambulance 108</a>
      </div>
      <form id="sosForm" novalidate>
        <label>What is happening?</label>
        <div class="cat-grid" id="sosCats">${categories.map((c) => `<button type="button" data-k="${c.key}" data-sev="${c.severity}"><span class="ic">${ICONS[c.key] || "❗"}</span>${A.esc(c.label)}</button>`).join("")}</div>
        <div class="field" style="margin-top:.9rem">
          <label for="sosDesc">Describe the situation</label>
          <textarea id="sosDesc" maxlength="2000" placeholder="What happened, who needs help, where in the bus you are…" required></textarea>
          <small id="sosCount">0 / 2000</small>
        </div>
        <div class="grid g2" style="margin-top:.6rem">
          <div><label for="sosRoute">Route no.</label><input id="sosRoute" placeholder="e.g. 500D" value="${A.esc(ctx.route_id || "")}"></div>
          <div><label for="sosBus">Bus</label><select id="sosBus"><option value="">Not sure</option></select></div>
        </div>
        <div class="grid g2" style="margin-top:.6rem">
          <div><label for="sosName">Your name</label><input id="sosName" value="${A.esc(u ? u.name : "")}" placeholder="Optional"></div>
          <div><label for="sosPhone">Phone for callback</label><input id="sosPhone" type="tel" value="${A.esc(u && u.phone ? u.phone : "")}" placeholder="Optional"></div>
        </div>
        <label class="check" style="margin-top:.8rem"><input type="checkbox" id="sosLoc" checked> Share my current location with the control room</label>
        <small id="sosLocState" class="muted"></small>
        <div id="sosErr" class="error" hidden style="margin-top:.7rem"></div>
        <button class="btn danger" style="width:100%;margin-top:1rem;padding:.8rem" id="sosSend" type="submit">Send emergency alert</button>
        <p class="muted" style="font-size:.8rem;margin:.6rem 0 0">Alerts go straight to the ATOBS control room. If anyone is in immediate danger, also call 112.</p>
      </form>`;
  }

  async function loadBuses(route) {
    const sel = body.querySelector("#sosBus");
    if (!sel) return;
    try {
      const all = (await A.api("/api/buses")).items;
      const list = route ? all.filter((b) => b.route_id.toLowerCase() === route.toLowerCase()) : all;
      const pre = (A.context || {}).bus_registration;
      sel.innerHTML = '<option value="">Not sure</option>' + list.map((b) => `<option ${b.registration === pre ? "selected" : ""} value="${A.esc(b.registration)}" data-route="${A.esc(b.route_id)}">${A.esc(b.registration)} · route ${A.esc(b.route_id)}</option>`).join("");
    } catch (_) { /* ignore */ }
  }

  function locate() {
    const st = body.querySelector("#sosLocState");
    coords = null;
    if (!body.querySelector("#sosLoc").checked) { st.textContent = ""; return; }
    if (!navigator.geolocation) { st.textContent = "Location isn't available in this browser."; return; }
    st.textContent = "Getting your location…";
    navigator.geolocation.getCurrentPosition(
      (p) => { coords = { lat: p.coords.latitude, lon: p.coords.longitude }; st.textContent = `Location attached (±${Math.round(p.coords.accuracy)} m)`; },
      () => { st.textContent = "Couldn't get location — the alert will still be sent."; },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  }

  function bind() {
    const form = body.querySelector("#sosForm");
    body.querySelectorAll("#sosCats button").forEach((b) =>
      b.addEventListener("click", () => {
        body.querySelectorAll("#sosCats button").forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        chosen = b.dataset.k;
      })
    );
    const desc = body.querySelector("#sosDesc");
    desc.addEventListener("input", () => (body.querySelector("#sosCount").textContent = `${desc.value.length} / 2000`));
    const route = body.querySelector("#sosRoute");
    route.addEventListener("change", () => loadBuses(route.value.trim()));
    body.querySelector("#sosBus").addEventListener("change", (e) => {
      const r = e.target.selectedOptions[0]?.dataset.route;
      if (r && !route.value) route.value = r;
    });
    body.querySelector("#sosLoc").addEventListener("change", locate);
    form.addEventListener("submit", submit);
    loadBuses(route.value.trim());
    locate();
    setTimeout(() => body.querySelector("#sosCats button")?.focus(), 50);
  }

  async function submit(e) {
    e.preventDefault();
    const err = body.querySelector("#sosErr");
    err.hidden = true;
    const desc = body.querySelector("#sosDesc").value.trim();
    if (!chosen) { err.textContent = "Choose what kind of emergency this is."; err.hidden = false; return; }
    if (desc.length < 5) { err.textContent = "Please describe the situation (at least a few words)."; err.hidden = false; return; }
    const btn = body.querySelector("#sosSend");
    btn.disabled = true;
    btn.textContent = "Sending…";
    const ctx = A.context || {};
    try {
      const r = await A.api("/api/emergency", {
        json: {
          category: chosen,
          description: desc,
          route_id: body.querySelector("#sosRoute").value.trim() || null,
          bus_registration: body.querySelector("#sosBus").value || null,
          stop_id: ctx.stop_id || null,
          reporter_name: body.querySelector("#sosName").value.trim() || null,
          reporter_phone: body.querySelector("#sosPhone").value.trim() || null,
          lat: coords ? coords.lat : null,
          lon: coords ? coords.lon : null,
          page_url: location.pathname + location.search,
        },
      });
      try {
        const saved = JSON.parse(localStorage.getItem("atobs_alerts") || "[]");
        saved.unshift(r.reference);
        localStorage.setItem("atobs_alerts", JSON.stringify(saved.slice(0, 20)));
      } catch (_) { /* storage unavailable */ }
      body.innerHTML = `<div class="sos-done">
        <div style="font-size:2.4rem">✅</div>
        <h2>Alert sent</h2>
        <p>${A.esc(r.message)} Keep this reference:</p>
        <div class="ref" id="sosRef">${A.esc(r.reference)}</div>
        <p class="muted">Priority: <b>${A.esc(r.severity)}</b>. Staff will update the status as they respond.</p>
        <div class="row" style="justify-content:center;margin-top:1rem">
          <a class="btn primary" href="/alerts.html?ref=${encodeURIComponent(r.reference)}">Track this alert</a>
          <a class="btn danger" href="tel:112">Call 112</a>
          <button class="btn" type="button" id="sosAgain">Close</button>
        </div></div>`;
      body.querySelector("#sosAgain").onclick = close;
    } catch (ex) {
      err.textContent = ex.message || "Couldn't send the alert. Call 112 now.";
      err.hidden = false;
      btn.disabled = false;
      btn.textContent = "Send emergency alert";
    }
  }

  async function open() {
    if (!categories.length) {
      try { categories = (await A.api("/api/emergency/categories")).items; } catch (_) { categories = []; }
    }
    try { await A.ready; } catch (_) { /* page guard */ }
    chosen = null;
    body.innerHTML = formHTML();
    bind();
    ov.classList.add("open");
  }
  function close() { ov.classList.remove("open"); fab.focus(); }

  fab.addEventListener("click", open);
  ov.querySelector("#sosClose").addEventListener("click", close);
  ov.addEventListener("click", (e) => { if (e.target === ov) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && ov.classList.contains("open")) close(); });
  A.openEmergency = open;
})();
