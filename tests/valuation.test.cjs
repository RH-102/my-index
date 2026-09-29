const test = require("node:test");
const assert = require("node:assert/strict");
const { render } = require("../valuation.js");

function rows() {
  return [
    { SourceId: "factset-sp500-ntm", Index: "S&P 500", ForwardPE: "19.2", Date: "2026-09-25", DateType: "报告日期", Kind: "未来12个月一致预期（周报）", Source: "FactSet", SourceURL: "https://example.com/report.pdf" },
    { SourceId: "hom-ndx-etf-fy1fy2", Index: "Nasdaq-100", ForwardPE: "21.35", Date: "2026-09-18", DateType: "观测日期", Kind: "ETF权重估算", Source: "History of Market", SourceURL: "https://historyofmarket.com/api/ndx/forward-pe.json" }
  ];
}
test("valuation cards distinguish reported consensus from estimated values and dates", () => {
  const html = render(rows(), new Date("2026-09-29T01:00:00Z"));
  for (const value of ["19.2×", "21.35×", "ETF权重估算", "报告日期：2026-09-25", "观测日期：2026-09-18"]) assert.ok(html.includes(value));
  assert.ok(!html.includes("来源观测超过14天"));
  assert.ok(render(rows(), new Date("2026-10-09T01:00:00Z")).includes("来源观测超过14天"));
});
test("missing values cannot render as a current valuation", () => {
  assert.throws(() => render(rows().slice(0, 1)));
  const bad = rows(); bad[0].ForwardPE = "";
  assert.throws(() => render(bad));
  bad[0].ForwardPE = "19.2"; bad[0].SourceURL = "javascript:alert(1)";
  assert.throws(() => render(bad));
});
