'use strict';
// Run: npm test (node --test). No Python engine needed.
const test = require('node:test');
const assert = require('node:assert');
const { createApp, sanitizeStageArgs } = require('../server');

test('sanitizeStageArgs accepts no args', () => {
  assert.deepStrictEqual(sanitizeStageArgs(undefined), { args: [] });
  assert.deepStrictEqual(sanitizeStageArgs([]), { args: [] });
});

test('sanitizeStageArgs accepts a well-formed --event and normalises it', () => {
  const r = sanitizeStageArgs(['--event', JSON.stringify({ ticker: 'aapl', date: '2026-09-15', extra: 'x' })]);
  assert.deepStrictEqual(r, { args: ['--event', '{"ticker":"AAPL","date":"2026-09-15"}'] });
});

test('sanitizeStageArgs rejects arbitrary engine flags', () => {
  for (const bad of [
    'nope',
    ['--tickers', 'AAPL'],
    ['--db', '/etc/passwd'],
    ['--event', '{"ticker":"AAPL","date":"2026-09-15"}', '--days', '9999'],
    ['--event', 'not json'],
    ['--event', '{"ticker":"$(rm)","date":"2026-09-15"}'],
    ['--event', '{"ticker":"AAPL","date":"yesterday"}'],
  ]) {
    assert.ok(sanitizeStageArgs(bad).error, `expected rejection for ${JSON.stringify(bad)}`);
  }
});

test('HTTP: health ok, unknown stage 400, unknown job 404', async () => {
  process.env.SIGNAL_LAB_MONITOR_MINUTES = '0';
  const server = createApp().listen(0);
  try {
    const base = `http://127.0.0.1:${server.address().port}`;
    const h = await fetch(`${base}/api/health`);
    assert.strictEqual(h.status, 200);
    assert.strictEqual((await h.json()).ok, true);
    const u = await fetch(`${base}/api/run/nope`, { method: 'POST' });
    assert.strictEqual(u.status, 400);
    const j = await fetch(`${base}/api/jobs/job-0`);
    assert.strictEqual(j.status, 404);
  } finally {
    server.close();
  }
});
