// 40-seat BMTC bus seat map (top view, front at the top, driver on the right).
// Green outline = free seat, red = occupied, standing passengers shown in the aisle.
//
//   ATOBS.seatMap(el, { occupied: 23 })                // fill 23 seats in a realistic order
//   ATOBS.seatMap(el, { seats: [1, 4, 5, ...] })       // exact seat numbers (e.g. simulation ground truth)
//   options: total (40), standing, seed (keeps the same seats filled for one bus), title, compact,
//            pct (show this occupancy % instead of seats/total, e.g. the model's forecast)
(function () {
  const A = (window.ATOBS = window.ATOBS || {});
  const ROWS = 10;
  const DOOR_ROWS = { 0: "Front door", 5: "Rear door" }; // door gaps on the left side, like BMTC buses

  function rng(seed) {
    let h = 2166136261;
    for (const c of String(seed)) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
    return () => {
      h = Math.imul(h ^ (h >>> 15), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      return ((h ^= h >>> 16) >>> 0) / 4294967296;
    };
  }

  // Order in which passengers usually take seats: window seats first, then aisle seats,
  // with some randomness (seeded so the same bus doesn't reshuffle on every refresh).
  function fillOrder(seed, total) {
    const r = rng(seed || "bus");
    const seats = [];
    for (let n = 1; n <= total; n++) {
      const col = (n - 1) % 4;
      const window = col === 0 || col === 3;
      seats.push({ n, key: (window ? 0 : 1) + r() * 0.9 });
    }
    return seats.sort((a, b) => a.key - b.key).map((s) => s.n);
  }

  A.seatMap = function (el, opt = {}) {
    const total = opt.total || 40;
    let taken;
    if (Array.isArray(opt.seats)) taken = new Set(opt.seats);
    else {
      const n = Math.max(0, Math.min(total, Math.round(opt.occupied || 0)));
      taken = new Set(fillOrder(opt.seed, total).slice(0, n));
    }
    const standing = Math.max(0, opt.standing || 0);
    const occ = taken.size;
    const pct = opt.pct != null ? Math.round(opt.pct) : Math.round((100 * (occ + standing)) / total);
    const seat = (n) => `<div class="seat ${taken.has(n) ? "taken" : "free"}" title="Seat ${n}: ${taken.has(n) ? "occupied" : "available"}">${n}</div>`;
    let rows = "";
    for (let r = 0; r < ROWS; r++) {
      const b = r * 4;
      const door = DOOR_ROWS[r] ? `<span class="door" title="${DOOR_ROWS[r]}"></span>` : "";
      rows += `<div class="seat-row">${door}${seat(b + 1)}${seat(b + 2)}<div class="aisle">${standingIn(r)}</div>${seat(b + 3)}${seat(b + 4)}</div>`;
    }
    function standingIn(r) {
      // spread standing passengers along the aisle (max 2 icons per row, rest summarised)
      const per = Math.ceil(standing / ROWS);
      const here = Math.max(0, Math.min(per, standing - r * per, 2));
      return "<i></i>".repeat(here);
    }
    el.innerHTML = `<div class="seatmap ${opt.compact ? "compact" : ""}">
      ${opt.title ? `<div class="sm-title">${opt.title}</div>` : ""}
      <div class="sm-summary"><b>${occ}</b> occupied · <b>${total - occ}</b> free${standing ? ` · <b>${standing}</b> standing` : ""} · <b class="sm-pct">${pct}%</b></div>
      <div class="bus-body">
        <div class="bus-front"><span>FRONT</span><svg class="wheel" viewBox="0 0 24 24" aria-label="Driver"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="2.5" fill="currentColor"/><path d="M2.5 11h7M14.5 11h7M12 14.5v7" stroke="currentColor" stroke-width="2"/></svg></div>
        ${rows}
      </div>
      <div class="sm-legend"><span class="lg free"></span>Available <span class="lg taken"></span>Occupied${standing ? ' <span class="lg stand"></span>Standing' : ""}</div>
    </div>`;
    return { occupied: occ, free: total - occ, standing, pct };
  };
})();
