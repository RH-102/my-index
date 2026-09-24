(function () {
  const names = { index: "指数", benchmarks: "基准", risk: "风险指标" };
  const maxAgeDays = {
    "Nasdaq-100 Forward P/E": 14, "S&P 500 Forward P/E": 14,
    "Nasdaq Cushion": 14, "S&P 500 Cushion": 14,
    "科技/AI估值风险": 14, "整体美股估值风险": 14,
    "DFII10 (10Y Real Yield)": 7, "HY OAS 3M Change": 7,
    "C&I SLOOS": 150, "Credit-to-GDP Gap": 210
  };

  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c]);
  }

  function easternDate(now = new Date()) {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit"
    }).formatToParts(now);
    const get = type => parts.find(part => part.type === type).value;
    return `${get("year")}-${get("month")}-${get("day")}`;
  }

  function riskDateNote(row, now = new Date()) {
    const limit = maxAgeDays[row.Indicator];
    if (limit === undefined) return "";
    const dates = String(row.DataDate || "").match(/\d{4}-\d{2}-\d{2}/g) || [];
    const date = dates.sort()[0];
    if (!date) return '<span class="data-warning">来源日期缺失，时效无法核验</span>';
    const age = Math.floor((Date.parse(easternDate(now)) - Date.parse(date)) / 86400000);
    const summaryDate = row.RowType === "Summary"
      ? `<span class="risk-small">评级依据截至 ${escape(date)}</span>` : "";
    return summaryDate + (age > limit
      ? `<span class="data-warning">来源观测已距今 ${age} 天（提醒阈值 ${limit} 天），请结合数据日期阅读。</span>`
      : "");
  }

  function render(report, now = new Date()) {
    if (!report.modules || !report.generated_at || !report.next_update_due_at) {
      throw new Error("Incomplete update status");
    }
    const overdue = now > new Date(report.next_update_due_at);
    const time = new Date(report.generated_at).toLocaleString("zh-CN", {
      timeZone: "America/New_York", hour12: false
    });
    const html = Object.entries(names).map(([key, label]) => {
      const module = report.modules[key] || {};
      let style = "", title = "更新正常";
      let detail = module.data_date ? `数据截至 ${module.data_date}` : "各项来源日期见下表";
      if (module.state === "failed") {
        style = "failed";
        title = "更新失败";
        detail += "；已保留上一份完整数据";
      } else if (module.state === "cached") {
        style = "warning";
        title = "部分来源刷新失败";
        detail += "；使用缓存数据";
      } else if (module.state !== "success" || !["current", "stale"].includes(module.freshness)) {
        style = "warning";
        title = "状态无法核验";
      }
      if (module.freshness === "stale") {
        style ||= "warning";
        if (title === "更新正常") title = key === "risk" ? "部分来源数据较旧" : "数据尚未跟上交易日";
        detail += key === "risk"
          ? `；${(module.indicators || []).filter(item => item.stale).length} 项超过提醒阈值`
          : `；应更新至 ${report.expected_market_date}`;
      }
      if (overdue) {
        style ||= "warning";
        if (title === "更新正常") title = "更新记录待刷新";
        detail += "；已超过下一交易日的预期更新时间";
      }
      return `<div class="status-card ${style}"><strong>${label} · ${escape(title)}</strong>` +
        `<span class="status-detail">${escape(detail)}</span>` +
        `<span class="status-detail">本次检查 ${escape(time)} ET</span></div>`;
    }).join("");
    document.getElementById("dataStatus").innerHTML = html;
  }

  async function load() {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch("data/update_status.json?v=" + Date.now(), {
        cache: "no-store", signal: controller.signal
      });
      if (!response.ok) throw new Error("Could not read update status");
      render(await response.json());
    } catch (error) {
      document.getElementById("dataStatus").innerHTML =
        '<div class="status-card warning"><strong>更新状态暂时无法读取</strong>' +
        '<span class="status-detail">请查看下方各项数据日期；刷新页面可重试。</span></div>';
      console.error(error);
    } finally {
      clearTimeout(timeout);
    }
  }

  window.DataStatus = { render, riskDateNote };
  load();
})();
