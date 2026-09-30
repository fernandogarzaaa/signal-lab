#!/usr/bin/env node
'use strict';
/**
 * signallab CLI.
 *
 *   signallab [demo] [--port N] [--no-open]   set up if needed, serve UI, open browser
 *   signallab run <stage>                     run one stage in the terminal
 *   signallab seed                            load the bundled demo snapshot
 *   signallab setup                           create engine venv + install deps
 *   signallab doctor                          environment checks
 *   signallab serve [--port N]                start the server only
 */
const path = require('path');
const {
  venvExists, setupEngine, seedDemo, doctor, ensureDataDir,
} = require('../lib/setup');
const { STAGES, STAGE_META, runStage } = require('../lib/jobs');
const { start } = require('../server');

const VERSION = require('../package.json').version;

function help() {
  console.log(`signallab ${VERSION} — news-sentiment market event engine

Usage:
  signallab [demo] [--port N] [--no-open]   set up, serve the UI, open browser
  signallab run <stage>                     run one stage in the terminal
  signallab seed                            load the bundled demo snapshot
  signallab setup                           create engine venv + install deps
  signallab doctor                          environment checks
  signallab serve [--port N] [--no-open]    start the server only

Stages: ${STAGES.join(' | ')}
Options: --port <n> (default 3100), --no-open, --help, --version`);
}

async function ensureSetup() {
  if (venvExists()) return true;
  console.log('First run: setting up the Python engine (one-time, a few minutes)...');
  try {
    await setupEngine((l) => console.log(l));
    return true;
  } catch (e) {
    console.error(`setup failed: ${e.message}`);
    return false;
  }
}

async function cmdDemo(port, openBrowser) {
  if (!(await ensureSetup())) process.exit(1);
  ensureDataDir();
  start(port, { openBrowser });
}

async function cmdRun(stage, extraArgs) {
  if (!STAGES.includes(stage)) {
    console.error(`unknown stage: ${stage}\nstages: ${STAGES.join(', ')}`);
    process.exit(1);
  }
  if (!(await ensureSetup())) process.exit(1);
  const meta = STAGE_META[stage];
  console.log(`[signallab] running stage: ${stage} — ${meta.title}`);
  const r = await runStage(stage, extraArgs, ({ stream, text }) => {
    (stream === 'err' ? process.stderr : process.stdout).write(text + '\n');
  });
  if (!r.ok) {
    console.error(`\n[signallab] stage failed: ${r.error}`);
    if (r.stderrTail) console.error('--- stderr tail ---\n' + r.stderrTail);
    process.exit(1);
  }
  console.log('\n[signallab] result:');
  console.log(JSON.stringify(r.result, null, 2));
}

function cmdDoctor() {
  const checks = doctor();
  let allPass = true;
  for (const c of checks) {
    const mark = c.pass ? 'PASS' : 'FAIL';
    if (!c.pass) allPass = false;
    console.log(`${mark}  ${c.name}\n       ${c.detail}`);
  }
  process.exit(allPass ? 0 : 1);
}

async function main() {
  const argv = process.argv.slice(2);
  let port = 3100, openBrowser = true;
  const args = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--port' && argv[i + 1]) { port = Number(argv[++i]); continue; }
    if (a === '--no-open') { openBrowser = false; continue; }
    if (a === '--help' || a === '-h') { help(); return; }
    if (a === '--version' || a === '-v') { console.log(VERSION); return; }
    args.push(a);
  }
  const [cmd, sub, ...rest] = args;
  if (!cmd || cmd === 'demo') return cmdDemo(port, openBrowser);
  if (cmd === 'serve') { ensureDataDir(); start(port, { openBrowser }); return; }
  if (cmd === 'setup') {
    try { await setupEngine((l) => console.log(l)); }
    catch (e) { console.error(`setup failed: ${e.message}`); process.exit(1); }
    return;
  }
  if (cmd === 'seed') {
    const copied = seedDemo();
    console.log(`seeded ${copied.length} files into data/`);
    copied.slice(0, 10).forEach((f) => console.log('  ' + f));
    if (copied.length > 10) console.log(`  ... and ${copied.length - 10} more`);
    return;
  }
  if (cmd === 'doctor') return cmdDoctor();
  if (cmd === 'run' && sub) return cmdRun(sub, rest);
  console.error(`unknown command: ${cmd}\n`);
  help();
  process.exit(1);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
