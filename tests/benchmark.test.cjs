const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function api() {
  const context = {window: {}, document: {getElementById: () => null}};
  // No timers or Chart.getChart are supplied: loading must be promise-driven.
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../benchmark-chart.js'), 'utf8'), context);
  return context.window.IndexBenchmark;
}
const rows = [{Date: '2026-08-12', Nasdaq100ReturnPct: 0, SP500ReturnPct: 0}];

test('comparison waits for a slow index without a five-second polling deadline', async () => {
  const benchmark = api();
  let resolveIndex;
  const index = new Promise(resolve => {resolveIndex = resolve;});
  let finished = false;
  const pending = benchmark.load(index, Promise.resolve(rows)).then(value => {finished = true; return value;});
  await new Promise(resolve => setTimeout(resolve, 6100));
  assert.equal(finished, false);
  resolveIndex(['holdings', 'history']);
  const result = await pending;
  assert.equal(result.benchmarkRows.length, 1);
  assert.equal(result.benchmarkError, null);
});

test('benchmark failure keeps index available with an explicit error', async () => {
  const result = await api().load(Promise.resolve(['holdings', 'history']), Promise.reject(new Error('HTTP 503')));
  assert.equal(result.indexData[0], 'holdings');
  assert.equal(result.benchmarkRows.length, 0);
  assert.match(result.benchmarkError.message, /503/);
});

test('missing numeric benchmark values are not silently turned into zero', () => {
  assert.throws(() => api().validateRows([{...rows[0], Nasdaq100ReturnPct: null}]), /malformed/);
});

test('retry adds both comparisons without converting index returns twice', () => {
  const benchmark = api();
  const chart = {
    data: {labels: ['2026-08-12'], datasets: [{data: [101]}]},
    options: {plugins: {legend: {}, tooltip: {callbacks: {}}}, scales: {y: {ticks: {}}}},
    update() {}
  };
  benchmark.apply(chart, []);
  benchmark.apply(chart, rows);
  assert.equal(chart.data.datasets[0].data[0], 1);
  assert.equal(chart.data.datasets.length, 3);
});
