/*
 * Does the reference's calibration hold across protein length and MSA depth?
 *
 * Two formulas are on trial, both explicitly calibrated in
 * sokrypton/laxy examples/gremlin_jax.ipynb:
 *
 *   lr    = 0.1 * log(B) / L                   -- shrinks with length
 *   lam_w = 0.5 * alpha * (L-1)(A-1) / Meff    -- grows with length, falls with depth
 *
 * Each is tested by sweeping a MULTIPLIER on it. A calibrated formula has its
 * best multiplier at ~1 everywhere; if the optimum drifts along the axis, the
 * exponent is off. Every protein is its own control -- multipliers are compared
 * within a protein, and only then is the argmax tracked across the axis.
 *
 * ---------------------------------------------------------------------------
 * Why length is varied by using different proteins, not by cropping
 * ---------------------------------------------------------------------------
 * The obvious design is to crop a window of columns. It does not work. A crop is
 * not a shorter protein, it is an amputated fragment: most of each residue's
 * contact partners fall outside the window, so the conditional for column i
 * loses most of its true predictors and the structure has almost nothing left to
 * score against. Measured on P0A9B2, a 40-column centred crop left 13 true
 * contacts among 630 pairs -- top-L/2 is then 4 hits out of 20 -- against 372
 * among 11935 for a whole protein at L=155. At that resolution a 16x change in
 * lr is invisible. Depth is still varied by subsampling, which genuinely does
 * produce a shallower alignment of the same protein.
 *
 * Usage (see ./README.md; run fetch-data.sh first):
 *   node sweep.mjs --arm=lr
 *   node sweep.mjs --arm=alpha --depths=128,512,2048,0 --mults=0.125,0.25,0.5,1,2
 *   node sweep.mjs --arm=alphaL --meff=400,1500          # length scaling, depth held fixed
 *   node sweep.mjs --arm=alpha --slice=0 --nslice=3      # run slices in parallel
 *
 * Appends JSON lines to results/<arm>.jsonl, skipping cells already present, so
 * a run is resumable and partial output is usable.
 */
import { readFileSync, appendFileSync, existsSync, mkdirSync } from 'node:fs';
import { parsePdb, cbContacts } from '../contacts.mjs';

const core = (await import('../../gremlin-core.js')).default;
const MSA = (await import('../../msa.js')).default;
const { Gremlin } = core;

const HERE = new URL('.', import.meta.url).pathname;
const WASM = HERE + '../../gremlin.wasm';

function opt(name, dflt) {
  const hit = process.argv.find(a => a.startsWith('--' + name + '='));
  return hit === undefined ? dflt : hit.slice(name.length + 3);
}
const ARM = opt('arm', 'lr');
const SLICE = parseInt(opt('slice', '0'), 10);
const NSLICE = parseInt(opt('nslice', '1'), 10);
const B = parseInt(opt('batch', '128'), 10);
const CHECK = opt('check', '100,200,400').split(',').map(Number);
const STEPS = Math.max(...CHECK);
const MINSEP = 5;

const manifest = JSON.parse(readFileSync(HERE + 'manifest.json', 'utf8'));
const ACCS = opt('proteins', '') ? opt('proteins', '').split(',')
                                 : (manifest[ARM] || manifest[ARM === 'lr' ? 'length' : 'depth']);
const MULTS = opt('mults', ARM === 'lr' ? '0.25,0.5,1,2,4' : '0.0625,0.125,0.25,0.5,1,2')
  .split(',').map(Number);
// 0 means "no subsampling"; the lr arm only ever runs at full depth
const DEPTHS = opt('depths', ARM === 'lr' ? '0' : '128,512,2048,0').split(',').map(Number);

mkdirSync(HERE + 'results', { recursive: true });
const OUT = HERE + 'results/' + ARM + '.jsonl';

/* ---- data ---- */
const cache = {};
function load(acc) {
  if (cache[acc]) return cache[acc];
  const a3m = HERE + 'data/' + acc + '.a3m', pdbf = HERE + 'data/' + acc + '.pdb';
  if (!existsSync(a3m) || !existsSync(pdbf)) {
    throw new Error('missing ' + acc + '; run ./fetch-data.sh');
  }
  const ds = MSA.buildDataset(readFileSync(a3m, 'utf8'), {
    keepQueryColumns: true, minCoverage: 0.75, minIdentity: 0.15, sortByIdentity: true
  });
  cache[acc] = { ds, isC: cbContacts(parsePdb(readFileSync(pdbf, 'utf8')), 8.0) };
  return cache[acc];
}

/*
 * How many rows to keep so that Meff lands near `target`.
 *
 * Needed for the length arm: the coupling penalty carries BOTH (L-1)(A-1) and
 * 1/Meff, so sweeping alpha across proteins of different length only measures
 * the length scaling if depth is held fixed. Deeper alignments would otherwise
 * masquerade as longer ones.
 *
 * Subsampling by a factor f thins each sequence's neighbourhood, so its
 * neighbour count goes from cnt_n to about 1 + f*(cnt_n - 1) and
 *   Meff(f) ~ f * sum_n 1/(1 + f*(cnt_n - 1)).
 * cnt_n is just 1/sw_n from the full-depth model, so this is a closed form and
 * the bisection costs nothing -- no model is built per probe. It only picks the
 * row count; the Meff actually recorded is the real one measured afterwards.
 */
function rowsForMeff(ds, sw, target) {
  const N = ds.N;
  const cnt = new Float64Array(N);
  for (let n = 0; n < N; n++) cnt[n] = 1 / sw[n];
  const est = (f) => {
    let s = 0;
    for (let n = 0; n < N; n++) s += 1 / (1 + f * (cnt[n] - 1));
    return f * s;
  };
  if (est(1) <= target) return 0;                 // not deep enough; use it all
  let lo = 1 / N, hi = 1;
  for (let it = 0; it < 40; it++) {
    const mid = Math.sqrt(lo * hi);
    if (est(mid) < target) lo = mid; else hi = mid;
  }
  return Math.max(8, Math.round(Math.sqrt(lo * hi) * N));
}

/* Deterministic subsample, query pinned first. rows <= 0 means keep everything. */
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
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      // separation in input-alignment columns, not model columns
      if (res[j] - res[i] < MINSEP) continue;
      const t = isC(res[i], res[j]);
      if (t !== null) out.push([cm[i * L + j], t]);
    }
  }
  out.sort((a, b) => b[0] - a[0]);
  const at = (k) => {
    const n = Math.min(k, out.length);
    if (!n) return null;
    let h = 0;
    for (let q = 0; q < n; q++) if (out[q][1]) h++;
    return h / n;
  };
  let nTrue = 0;
  for (const o of out) if (o[1]) nTrue++;
  return { l5: at(Math.floor(L / 5)), l2: at(Math.floor(L / 2)), l: at(L), nTrue, nPairs: out.length };
}

const done = new Set();
if (existsSync(OUT)) {
  for (const ln of readFileSync(OUT, 'utf8').split('\n')) {
    if (!ln.trim()) continue;
    try { done.add(JSON.parse(ln).key); } catch (e) { /* truncated tail */ }
  }
}

/*
 * The length arm targets a Meff instead of a row count, so proteins of
 * different length are compared at matched depth. Row counts are resolved
 * lazily, since it needs each protein's full-depth weights.
 */
const MEFFS = opt('meff', '') ? opt('meff', '').split(',').map(Number) : null;
const fullSw = {};        // full-depth sequence weights, one pass per protein
async function rowsFor(acc, target) {
  const { ds } = load(acc);
  if (!fullSw[acc]) {
    const wasm = await core.initWasm(WASM);
    // built only for its reweighting pass; the parameters are never touched,
    // and it is dropped straight after so the allocation does not accumulate
    const probe = new Gremlin({
      L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, backend: wasm,
      identity: 0.8, seed: 1, cfg: { batch: 8 }
    });
    fullSw[acc] = Float32Array.from(probe.sw);
  }
  return rowsForMeff(ds, fullSw[acc], target);
}

const JOBS = [];
if (MEFFS) {
  for (const acc of ACCS) for (const target of MEFFS) for (const m of MULTS) {
    JOBS.push({ acc, target, rows: -1, m });
  }
} else {
  for (const acc of ACCS) for (const rows of DEPTHS) for (const m of MULTS) JOBS.push({ acc, rows, m });
}
const mine = JOBS.filter((_, i) => i % NSLICE === SLICE);
process.stderr.write('arm=' + ARM + ' slice ' + SLICE + '/' + NSLICE
  + ': ' + mine.length + ' cells, ' + done.size + ' already done\n');

let i = 0;
for (const job of mine) {
  i++;
  const key = [job.acc, job.target === undefined ? job.rows : 'M' + job.target, job.m].join('/');
  if (done.has(key)) continue;
  const { ds, isC } = load(job.acc);
  const rows = job.target === undefined ? job.rows : await rowsFor(job.acc, job.target);
  const sub = subsample(ds, rows, 99);
  const lrMult = ARM === 'lr' ? job.m : 1;
  const aMult = ARM === 'lr' ? 1 : job.m;

  // a fresh instance per fit: the WASM backend bump-allocates and never frees
  const wasm = await core.initWasm(WASM);
  const t0 = Date.now();
  const g = new Gremlin({
    L: ds.L, A: ds.A, N: sub.N, seqs: sub.seqs, backend: wasm,
    gap: 20, biasInit: 'freq', identity: 0.8, seed: 1234567,
    cfg: { batch: B, lr: lrMult * Gremlin.suggestLr(ds.L, B),
           alpha: 0.01 * aMult, beta: 0.01, regMode: 'gremlin' }
  });
  const at = {};
  let step = 0;
  for (const ck of CHECK) {
    while (step < ck) { g.step(); step++; }
    at[ck] = precision(g.contactMap(), ds.L, ds.colMap, isC);
  }
  appendFileSync(OUT, JSON.stringify({
    key, acc: job.acc, arm: ARM, lrMult, aMult, rows,
    target: job.target === undefined ? null : job.target,
    L: ds.L, N: sub.N, Meff: g.Meff, at, secs: (Date.now() - t0) / 1000
  }) + '\n');
  process.stderr.write('[' + i + '/' + mine.length + '] ' + key
    + '  L=' + ds.L + ' N=' + sub.N + ' Meff=' + g.Meff.toFixed(0)
    + '  L/2@' + STEPS + '=' + (at[STEPS].l2 * 100).toFixed(1) + '%'
    + '  nTrue=' + at[STEPS].nTrue + '  ' + ((Date.now() - t0) / 1000).toFixed(0) + 's\n');
}
process.stderr.write('slice ' + SLICE + ' done\n');
