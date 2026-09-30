'use strict';
/**
 * signallab web server: serves the dashboard UI and the stage-runner API.
 *
 *   node server.js [--port N]
 *
 * API:
 *   GET  /api/status            engine + per-stage run state
 *   POST /api/run/:stage        start a stage job  {job_id, stage, status}
 *   GET  /api/jobs/:jobid       job status {status, error, has_result}
 *   GET  /api/logs/:jobid       Server-Sent Events stream of job output
 *   GET  /api/results/:stage    cached JSON or 404 {"error":"not run yet"}
 *   POST /api/seed              load the bundled demo snapshot
 */
const path = require('path');
const fs = require('fs');
const express = require('express');
const {
  PKG_ROOT, DATA_DIR, RESULTS_DIR, VENV_DIR,
  venvExists, python3Version, ensureDataDir, seedDemo, doctor,
} = require('./lib/setup');
const { createJobManager, STAGES, STAGE_META } = require('./lib/jobs');

const PUBLIC_DIR = path.join(PKG_ROOT, 'public');

function stageState(stage) {
  const f = path.join(RESULTS_DIR, `${stage}.json`);
  try {
    const st = fs.statSync(f);
    return { ran: true, at: st.mtime.toISOString() };
  } catch {
    return { ran: false, at: null };
  }
}

function createApp() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));
  app.use(express.static(PUBLIC_DIR));

  const jobs = createJobManager();

  app.get('/api/status', (req, res) => {
    const stages = {};
    for (const s of STAGES) stages[s] = { ...STAGE_META[s] && { title: STAGE_META[s].title }, ...stageState(s) };
    res.json({
      engine: venvExists() ? 'ok' : 'missing',
      venv: VENV_DIR,
      python: python3Version(),
      node: process.version,
      stages,
    });
  });

  app.get('/api/doctor', (req, res) => res.json({ checks: doctor() }));

  app.post('/api/run/:stage', (req, res) => {
    const { stage } = req.params;
    if (!STAGES.includes(stage)) return res.status(400).json({ error: `unknown stage: ${stage}` });
    if (!venvExists()) {
      return res.status(400).json({ error: 'engine not set up — run `signallab setup` first' });
    }
    const extraArgs = Array.isArray(req.body && req.body.args) ? req.body.args.map(String) : [];
    const job = jobs.start(stage, extraArgs);
    res.json({ job_id: job.id, stage, status: job.status });
  });

  app.get('/api/jobs/:jobid', (req, res) => {
    const job = jobs.get(req.params.jobid);
    if (!job) return res.status(404).json({ error: 'unknown job' });
    res.json(jobs.summary(job));
  });

  // Server-Sent Events: replay buffered lines, then stream until job ends.
  app.get('/api/logs/:jobid', (req, res) => {
    const job = jobs.get(req.params.jobid);
    if (!job) return res.status(404).json({ error: 'unknown job' });
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });
    const sendLine = (l) => res.write(`data: ${JSON.stringify({ t: l.t, s: l.s, text: l.text })}\n\n`);
    for (const l of job.lines) sendLine(l);
    const onLine = (l) => sendLine(l);
    const onEnd = () => {
      res.write(`event: done\ndata: ${JSON.stringify(jobs.summary(job))}\n\n`);
      cleanup();
    };
    const cleanup = () => {
      job.emitter.removeListener('line', onLine);
      job.emitter.removeListener('end', onEnd);
      try { res.end(); } catch {}
    };
    if (job.status !== 'running') { onEnd(); return; }
    job.emitter.on('line', onLine);
    job.emitter.on('end', onEnd);
    req.on('close', cleanup);
  });

  app.get('/api/results/:stage', (req, res) => {
    const { stage } = req.params;
    if (!STAGES.includes(stage)) return res.status(400).json({ error: `unknown stage: ${stage}` });
    const f = path.join(RESULTS_DIR, `${stage}.json`);
    try {
      res.json(JSON.parse(fs.readFileSync(f, 'utf8')));
    } catch {
      res.status(404).json({ error: 'not run yet' });
    }
  });

  app.post('/api/seed', (req, res) => {
    try {
      const copied = seedDemo();
      res.json({ ok: true, copied });
    } catch (e) {
      res.status(500).json({ ok: false, error: e.message });
    }
  });

  // Fallback to the dashboard for any other route.
  app.get('*', (req, res) => res.sendFile(path.join(PUBLIC_DIR, 'index.html')));

  return app;
}

function start(port, { openBrowser = true } = {}) {
  const app = createApp();
  const server = app.listen(port, () => {
    const url = `http://localhost:${port}`;
    console.log(`Signal Lab UI: ${url}`);
    if (openBrowser) {
      import('open').then((m) => m.default(url).catch((e) => {
        console.log(`(could not open a browser automatically: ${e.message})`);
        console.log(`Open this URL manually: ${url}`);
      })).catch(() => console.log(`Open this URL manually: ${url}`));
    }
  });
  return server;
}

if (require.main === module) {
  const args = process.argv.slice(2);
  let port = 3100;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' && args[i + 1]) port = Number(args[++i]);
  }
  ensureDataDir();
  start(port, { openBrowser: false });
  console.log('(server.js run directly; the signallab CLI opens the browser for you)');
}

module.exports = { createApp, start };
