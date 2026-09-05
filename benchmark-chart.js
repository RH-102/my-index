(function () {
  const BENCHMARK_FILE = "data/benchmark_history.csv";

  function formatQuantityDecimals() {
    const tableArea = document.getElementById("tableArea");
    if (!tableArea) return false;

    const table = tableArea.querySelector("table");
    if (!table) return false;

    const headers = Array.from(table.querySelectorAll("thead th"));
    const quantityIndex = headers.findIndex(
      th => th.textContent.trim() === "Quantity"
    );

    if (quantityIndex < 0) return false;

    table.querySelectorAll("tbody tr").forEach(row => {
      const cell = row.children[quantityIndex];
      if (!cell) return;

      const value = Number(cell.textContent.replace(/,/g, "").trim());
      if (Number.isFinite(value)) {
        cell.textContent = value.toFixed(4);
      }
    });

    return true;
  }

  function watchQuantityTable() {
    if (formatQuantityDecimals()) return;

    const tableArea = document.getElementById("tableArea");
    if (!tableArea) return;

    const observer = new MutationObserver(() => {
      if (formatQuantityDecimals()) {
        observer.disconnect();
      }
    });

    observer.observe(tableArea, {
      childList: true,
      subtree: true
    });
  }

  function loadBenchmarkCSV() {
    const url = BENCHMARK_FILE + "?v=" + Date.now();
    return new Promise((resolve) => {
      Papa.parse(url, {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: results => resolve(
          results.data.filter(row => row.Date)
        ),
        error: () => resolve([])
      });
    });
  }

  function signedPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return (number > 0 ? "+" : "") + number.toFixed(2) + "%";
  }

  function updateHeading() {
    const canvas = document.getElementById("indexChart");
    const section = canvas && canvas.closest(".section");
    if (!section) return;

    const heading = section.querySelector("h2");
    if (heading) heading.textContent = "Index Performance (% Return)";

    if (!section.querySelector(".benchmark-subtitle")) {
      const subtitle = document.createElement("div");
      subtitle.className = "subtitle benchmark-subtitle";
      subtitle.textContent =
        "2026-08-12 closing level = 100 for My Index, Nasdaq-100 and S&P 500; chart shows cumulative return since that close.";
      if (heading) heading.insertAdjacentElement("afterend", subtitle);
    }
  }

  function applyComparison(benchmarkRows) {
    const chart = Chart.getChart("indexChart");
    if (!chart) return false;

    const benchmarkMap = new Map(
      benchmarkRows.map(row => [String(row.Date), row])
    );

    const labels = chart.data.labels.map(value => String(value));
    const original = chart.data.datasets[0];

    if (!original._convertedToReturn) {
      original.data = original.data.map(value => Number(value) - 100.0);
      original.label = "My Index";
      original._convertedToReturn = true;
    }

    // Translucent, visually distinct comparison colors.
    original.borderColor = "rgba(107, 114, 128, 0.72)";
    original.backgroundColor = "rgba(107, 114, 128, 0.18)";
    original.pointBackgroundColor = "rgba(107, 114, 128, 0.88)";
    original.pointBorderColor = "rgba(107, 114, 128, 0.88)";

    chart.data.datasets = [original];

    if (benchmarkRows.length > 0) {
      chart.data.datasets.push(
        {
          label: "Nasdaq-100",
          data: labels.map(date => {
            const row = benchmarkMap.get(date);
            return row ? Number(row.Nasdaq100ReturnPct) : null;
          }),
          tension: 0.15,
          pointRadius: 3,
          borderWidth: 2,
          spanGaps: false,
          borderColor: "rgba(37, 99, 235, 0.72)",
          backgroundColor: "rgba(37, 99, 235, 0.18)",
          pointBackgroundColor: "rgba(37, 99, 235, 0.88)",
          pointBorderColor: "rgba(37, 99, 235, 0.88)"
        },
        {
          label: "S&P 500",
          data: labels.map(date => {
            const row = benchmarkMap.get(date);
            return row ? Number(row.SP500ReturnPct) : null;
          }),
          tension: 0.15,
          pointRadius: 3,
          borderWidth: 2,
          spanGaps: false,
          borderColor: "rgba(22, 163, 74, 0.72)",
          backgroundColor: "rgba(22, 163, 74, 0.18)",
          pointBackgroundColor: "rgba(22, 163, 74, 0.88)",
          pointBorderColor: "rgba(22, 163, 74, 0.88)"
        }
      );
    }

    chart.options.plugins.legend.display = true;
    chart.options.plugins.tooltip.callbacks.label = context =>
      context.dataset.label + ": " + signedPercent(context.raw);

    chart.options.scales.y.ticks.callback = value => signedPercent(value);
    chart.options.scales.y.title = {
      display: true,
      text: "Cumulative return since 2026-08-12"
    };

    chart.update();
    return true;
  }

  async function init() {
    watchQuantityTable();
    updateHeading();
    const benchmarkRows = await loadBenchmarkCSV();

    let attempts = 0;
    const timer = setInterval(() => {
      attempts += 1;
      if (applyComparison(benchmarkRows) || attempts >= 50) {
        clearInterval(timer);
      }
    }, 100);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
