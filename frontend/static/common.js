// Shared helpers for every dashboard page: nav bar, API calls, toasts, settings, theme.
(() => {
  const PAGES = [["/", "Live"], ["/record", "Record & Train"], ["/test", "Self-test"],
                 ["/history", "History"], ["/settings", "Settings"]];

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers["Content-Type"] = "application/json"; }
    const r = await fetch(path, init);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`);
    return data;
  }

  function toast(msg, kind = "") {
    let box = document.getElementById("toasts");
    if (!box) { box = document.createElement("div"); box.id = "toasts"; document.body.appendChild(box); }
    const t = document.createElement("div"); t.className = "toast " + kind; t.textContent = msg;
    box.appendChild(t); setTimeout(() => t.remove(), 4500);
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
    try { localStorage.setItem("jsl.theme", theme); } catch (e) {}
  }
  try { applyTheme(localStorage.getItem("jsl.theme") || "dark"); } catch (e) {}

  function nav(extraRight = "") {
    const el = document.createElement("nav"); el.className = "nav";
    const here = location.pathname.replace(/\/$/, "") || "/";
    el.innerHTML = `<div class="brand">JSL Recognition <span>· 日本手話 · University of Aizu</span></div>` +
      PAGES.map(([href, label]) => `<a class="tab${href === here ? " active" : ""}" href="${href}">${label}</a>`).join("") +
      `<div class="right">${extraRight}</div>`;
    document.body.prepend(el);
    return el;
  }

  let cachedSettings = null;
  async function settings(force = false) {
    if (!cachedSettings || force) {
      cachedSettings = await api("/api/settings");
      applyTheme(cachedSettings.settings.display.theme);
    }
    return cachedSettings;
  }

  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = v => v == null ? "—" : Math.round(v * 100) + "%";
  const time = ts => new Date(ts * 1000).toLocaleString();

  window.JSL = { api, toast, nav, settings, applyTheme, esc, pct, time };
})();
