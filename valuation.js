(function (root) {
  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c]);
  }

  function render(rows, now = new Date()) {
    const expected = ["factset-sp500-ntm", "hom-ndx-etf-fy1fy2"];
    const today = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(now);
    return expected.map(id => {
      const matches = rows.filter(row => row.SourceId === id);
      if (matches.length !== 1) throw new Error("Missing or duplicate valuation source");
      const row = matches[0];
      const value = Number(row.ForwardPE);
      const date = String(row.Date || "");
      const url = new URL(row.SourceURL);
      if (!Number.isFinite(value) || value <= 1 || value >= 100 ||
          !/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) ||
          date > today || url.protocol !== "https:") {
        throw new Error("Invalid valuation reference");
      }
      const age = (Date.parse(today) - Date.parse(date)) / 86400000;
      const warning = age > 14 ? '<span class="data-warning">来源观测超过14天，等待新数据</span>' : "";
      const formatted = value.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 2 });
      return `<div class="risk-summary-card">
        <div class="risk-summary-title">${escape(row.Index)} Forward P/E</div>
        <div class="risk-summary-value">${formatted}×</div>
        <span class="risk-small">${escape(row.Kind)}</span>
        <span class="risk-small">${escape(row.DateType)}：${escape(date)}</span>
        ${warning}
        <a class="risk-small" href="${escape(url.href)}" target="_blank" rel="noopener noreferrer">${escape(row.Source)} ↗</a>
        <p class="risk-small">${escape(row.Method)}</p>
      </div>`;
    }).join("");
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { render };
  root.Valuation = { render };
})(typeof window === "undefined" ? globalThis : window);
