/*
 * Head-to-head on held-out proteins: does the depth correction actually beat
 * plain 1/Meff, measured rather than predicted?
 *
 * The gain numbers in refit.mjs come from evaluating the fitted parabola at its
 * own vertex against the same parabola at m = 1. That is circular twice over --
 * fitted on this data, and read off the model instead of the measurement. This
 * runs the formulas head to head on proteins that were NOT in the fit, and
 * reports the paired difference.
 *
 * Three settings, because the fit and the proposal are not the same thing:
 *   ref    m = 1                          the reference, lam_w ~ 1/Meff
 *   expo   m = (700/Meff)^0.44            corrected exponent, alpha unchanged
 *                                         at the pin -- what the README proposed
 *   full   m = 0.688*(700/Meff)^0.44      the actual fitted optimum, which also
 *                                         lowers alpha at the pin
 *
 * Paired by construction: same protein, same subsample, same seed, same steps;
 * only alpha differs.
 *
 *   node holdout.mjs [--slice=0 --nslice=3]
 */
import { readFileSync, appendFileSync, existsSync, mkdirSync } from 'node:fs';
import { parsePdb, cbContacts } from '../contacts.mjs';

const core = (await import('../../gremlin-core.js')).default;
const MSA = (await import('../../msa.js')).default;
const { Gremlin } = core;

const HERE = new URL('.', import.meta.url).pathname;
const WASM = HERE + '../../gremlin.wasm';
function opt(n, d) {
  const h = process.argv.find(a => a.startsWith('--' + n + '='));
  return h === undefined ? d : h.slice(n.length + 3);
}
const SLICE = parseInt(opt('slice', '0'), 10);
const NSLICE = parseInt(opt('nslice', '1'), 10);
const B = 128, STEPS = 400, MINSEP = 5, MREF = 700;

const HOLDOUT = JSON.parse(readFileSync(HERE + 'manifest.json', 'utf8')).holdout;
const SETTINGS = {
  ref:  () => 1,
  expo: (meff) => Math.pow(MREF / meff, 0.44),
  full: (meff) => 0.688 * Math.pow(MREF / meff, 0.44)
};
const DEPTHS = [150, 0];          // shallow (correction raises alpha) and full

mkdirSync(HERE + 'results', { recursive: true });
const OUT = HERE + 'results/holdout.jsonl';

const cache = {};
function load(acc) {
  if (cache[acc]) return cache[acc];
  const ds = MSA.buildDataset(readFileSync(HERE + 'data/' + acc + '.a3m', 'utf8'), {
    keepQueryColumns: true, minCoverage: 0.75, minIdentity: 0.15, sortByIdentity: true
  });
  cache[acc] = { ds, isC: cbContacts(parsePdb(readFileSync(HERE + 'data/' + acc + '.pdb', 'utf8')), 8.0) };
  return cache[acc];
}
function subsample(ds, rows, seed) {
  if (!(rows > 0) || rows >= ds.N) return { N: ds.N, seqs: ds.seqs };
  let st = (seed || 1) >>> 0;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const pool = [];
  for (let n = 1; n < ds.N; n++) pool.push(n);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const t = pool[i]; pool[i] = pool[j]; pool[j] = t;
  }
  const pick = [0].concat(pool.slice(0, rows - 1));
  const seqs = new Int32Array(pick.length * ds.L);
  for (let q = 0; q < pick.length; q++) {
    seqs.set(ds.seqs.subarray(pick[q] * ds.L, (pick[q] + 1) * ds.L), q * ds.L);
  }
  return { N: pick.length, seqs };
}
function precision(cm, L, res, isC) {
  const out = [];
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) {
    if (res[j] - res[i] < MINSEP) continue;
    const t = isC(res[i], res[j]);
    if (t !== null) out.push([cm[i * L + j], t]);
  }
  out.sort((a, b) => b[0] - a[0]);
  const at = (k) => {
    const n = Math.min(k, out.length);
    if (!n) return null;
    let h = 0;
    for (let q = 0; q < n; q++) if (out[q][1]) h++;
    return h / n;
  };
  return { l5: at(Math.floor(L / 5)), l2: at(Math.floor(L / 2)), l: at(L) };
}

const done = new Set();
if (existsSync(OUT)) {
  for (const ln of readFileSync(OUT, 'utf8').split('\n')) {
    if (!ln.trim()) continue;
    try { done.add(JSON.parse(ln).key); } catch (e) { /* truncated */ }
  }
}

const JOBS = [];
for (const acc of HOLDOUT) for (const rows of DEPTHS) for (const s of Object.keys(SETTINGS)) {
  JOBS.push({ acc, rows, s });
}
const mine = JOBS.filter((_, i) => i % NSLICE === SLICE);
process.stderr.write('holdout slice ' + SLICE + ': ' + mine.length + ' cells\n');

let i = 0;
for (const job of mine) {
  i++;
  const key = [job.acc, job.rows, job.s].join('/');
  if (done.has(key)) continue;
  const { ds, isC } = load(job.acc);
  const sub = subsample(ds, job.rows, 99);
  // Meff has to be known before alpha can be set, so build once to read it and
  // once to train. The reweighting pass is the cheap part.
  const w0 = await core.initWasm(WASM);
  const probe = new Gremlin({
    L: ds.L, A: ds.A, N: sub.N, seqs: sub.seqs, backend: w0,
    identity: 0.8, seed: 1234567, cfg: { batch: 8 }
  });
  const meff = probe.Meff;
  const mult = SETTINGS[job.s](meff);

  const wasm = await core.initWasm(WASM);
  const t0 = Date.now();
  const g = new Gremlin({
    L: ds.L, A: ds.A, N: sub.N, seqs: sub.seqs, backend: wasm,
    gap: 20, biasInit: 'freq', identity: 0.8, seed: 1234567,
    cfg: { batch: B, lr: Gremlin.suggestLr(ds.L, B), alpha: 0.01 * mult,
           beta: 0.01, regMode: 'gremlin' }
  });
  for (let s = 0; s < STEPS; s++) g.step();
  const p = precision(g.contactMap(), ds.L, ds.colMap, isC);
  appendFileSync(OUT, JSON.stringify({
    key, acc: job.acc, setting: job.s, rows: job.rows, L: ds.L, N: sub.N,
    Meff: meff, mult, p, secs: (Date.now() - t0) / 1000
  }) + '\n');
  process.stderr.write('[' + i + '/' + mine.length + '] ' + key
    + '  L=' + ds.L + ' Meff=' + meff.toFixed(0) + ' m=' + mult.toFixed(3)
    + '  L/2=' + (p.l2 * 100).toFixed(1) + '%  ' + ((Date.now() - t0) / 1000).toFixed(0) + 's\n');
}
process.stderr.write('slice ' + SLICE + ' done\n');
