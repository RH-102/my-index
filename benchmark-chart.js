(function () {
  const BENCHMARK_FILE = "data/benchmark_history.csv";

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

    // Keep all three comparison lines visually distinct.
    original.borderColor = "#111827";
    original.backgroundColor = "#111827";
    original.pointBackgroundColor = "#111827";
    original.pointBorderColor = "#111827";

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
          borderColor: "#2563eb",
          backgroundColor: "#2563eb",
          pointBackgroundColor: "#2563eb",
          pointBorderColor: "#2563eb"
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
          borderColor: "#f59e0b",
          backgroundColor: "#f59e0b",
          pointBackgroundColor: "#f59e0b",
          pointBorderColor: "#f59e0b"
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
