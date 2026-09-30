'use strict';
/**
 * Shared paths and lifecycle helpers: setup (venv), doctor checks, seed.
 */
const path = require('path');
const fs = require('fs');
const { execFileSync, spawnSync } = require('child_process');

const PKG_ROOT = path.resolve(__dirname, '..');
const BIN_DIR = path.join(PKG_ROOT, 'bin');
const ENGINE_DIR = path.join(PKG_ROOT, 'engine');
const ENGINE_PKG = path.join(ENGINE_DIR, 'signal_lab');
const ENGINE_REQ = path.join(ENGINE_DIR, 'requirements.txt');
const SEED_DIR = path.join(ENGINE_DIR, 'seed_data');
const ADAPTERS_DIR = path.join(PKG_ROOT, 'adapters');
const VENV_DIR = path.join(ENGINE_DIR, '.venv');
const DATA_DIR = path.join(PKG_ROOT, 'data');
const RESULTS_DIR = path.join(DATA_DIR, 'results');

function venvPython() {
  const p = process.platform === 'win32'
    ? path.join(VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VENV_DIR, 'bin', 'python');
  return p;
}

function venvExists() {
  try {
    return fs.existsSync(venvPython()) && fs.statSync(venvPython()).isFile();
  } catch { return false; }
}

function python3Version() {
  try {
    const r = spawnSync('python3', ['--version'], { encoding: 'utf8' });
    if (r.status === 0) return (r.stdout || r.stderr || '').trim();
  } catch {}
  return null;
}

function ensureDataDir() {
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  // Engine seed files the Python code expects under <root>/data.
  const seeds = ['aliases.csv', 'labels.csv'];
  for (const f of seeds) {
    const dst = path.join(DATA_DIR, f);
    const src = path.join(SEED_DIR, f);
    if (!fs.existsSync(dst) && fs.existsSync(src)) {
      fs.copyFileSync(src, dst);
    }
  }
}

/**
 * Create engine/.venv and pip-install engine/requirements.txt.
 * onLine(line) receives progress lines. Resolves on success, rejects on failure.
 */
function setupEngine(onLine) {
  return new Promise((resolve, reject) => {
    const say = (l) => { try { onLine && onLine(l); } catch {} };
    if (!fs.existsSync(ENGINE_PKG)) {
      return reject(new Error(`engine package missing: ${ENGINE_PKG}`));
    }
    if (!fs.existsSync(ENGINE_REQ)) {
      return reject(new Error(`engine requirements missing: ${ENGINE_REQ}`));
    }
    const steps = [];
    if (!venvExists()) {
      steps.push({ name: 'creating venv', cmd: 'python3', args: ['-m', 'venv', VENV_DIR] });
    } else {
      say('[setup] venv already exists, skipping creation');
    }
    steps.push({ name: 'installing requirements', cmd: venvPython(), args: ['-m', 'pip', 'install', '--quiet', '--upgrade', 'pip'] });
    steps.push({ name: 'installing engine deps', cmd: venvPython(), args: ['-m', 'pip', 'install', '--quiet', '-r', ENGINE_REQ] });

    const runStep = (i) => {
      if (i >= steps.length) {
        ensureDataDir();
        say('[setup] done');
        return resolve();
      }
      const s = steps[i];
      say(`[setup] ${s.name} ...`);
      const child = require('child_process').spawn(s.cmd, s.args, { cwd: PKG_ROOT });
      child.stdout.on('data', (d) => String(d).split('\n').forEach((l) => l.trim() && say('  ' + l.trim())));
      child.stderr.on('data', (d) => String(d).split('\n').forEach((l) => l.trim() && say('  ' + l.trim())));
      child.on('error', (e) => reject(new Error(`[setup] ${s.name} failed to start: ${e.message}`)));
      child.on('close', (code) => {
        if (code !== 0) return reject(new Error(`[setup] ${s.name} exited with code ${code}`));
        say(`[setup] ${s.name}: ok`);
        runStep(i + 1);
      });
    };
    runStep(0);
  });
}

/** Copy the bundled demo snapshot into data/. Returns list of copied files. */
function seedDemo() {
  ensureDataDir();
  const copied = [];
  const copyTree = (src, dst) => {
    if (!fs.existsSync(src)) return;
    const st = fs.statSync(src);
    if (st.isDirectory()) {
      fs.mkdirSync(dst, { recursive: true });
      for (const e of fs.readdirSync(src)) copyTree(path.join(src, e), path.join(dst, e));
    } else {
      if (path.basename(src) === '.venv') return;
      fs.copyFileSync(src, dst);
      copied.push(path.relative(PKG_ROOT, dst));
    }
  };
  for (const name of ['signal_lab.duckdb', 'aliases.csv', 'labels.csv', 'artifacts']) {
    copyTree(path.join(SEED_DIR, name), path.join(DATA_DIR, name));
  }
  // Precomputed stage results (from a real run) render visuals instantly.
  copyTree(path.join(SEED_DIR, 'results'), RESULTS_DIR);
  return copied;
}

/** Doctor checks. Returns [{name, pass, detail}]. */
function doctor() {
  const checks = [];
  const nodeOk = Number(process.versions.node.split('.')[0]) >= 18;
  checks.push({ name: 'node >= 18', pass: nodeOk, detail: process.version });

  const pyv = python3Version();
  checks.push({ name: 'python3 available', pass: !!pyv, detail: pyv || 'not found on PATH' });

  const venvOk = venvExists();
  checks.push({
    name: 'engine venv', pass: venvOk,
    detail: venvOk ? venvPython() : 'missing — run `signallab setup`',
  });

  let engineImport = false, engineDetail = 'venv missing';
  if (venvOk) {
    try {
      const r = spawnSync(venvPython(), ['-c', 'import signal_lab; print(signal_lab.__file__)'],
        { encoding: 'utf8', env: { ...process.env, PYTHONPATH: ENGINE_DIR } });
      engineImport = r.status === 0;
      engineDetail = engineImport ? (r.stdout || '').trim() : 'import failed: ' + ((r.stderr || '').trim().split('\n').slice(-2).join(' '));
    } catch (e) { engineDetail = e.message; }
  }
  checks.push({ name: 'engine importable', pass: engineImport, detail: engineDetail });

  let dataOk = false, dataDetail = '';
  try {
    ensureDataDir();
    const probe = path.join(DATA_DIR, '.writetest');
    fs.writeFileSync(probe, 'ok');
    fs.unlinkSync(probe);
    dataOk = true; dataDetail = DATA_DIR;
  } catch (e) { dataDetail = e.message; }
  checks.push({ name: 'data dir writable', pass: dataOk, detail: dataDetail });

  return checks;
}

module.exports = {
  PKG_ROOT, ENGINE_DIR, ENGINE_PKG, ENGINE_REQ, SEED_DIR, ADAPTERS_DIR,
  VENV_DIR, DATA_DIR, RESULTS_DIR,
  venvPython, venvExists, python3Version, ensureDataDir,
  setupEngine, seedDemo, doctor,
};
