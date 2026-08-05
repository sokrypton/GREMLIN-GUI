/*
 * node test/core.test.mjs
 *
 * 1. the rewritten gradient matches the original implementation exactly
 * 2. W stays bit-exactly symmetric across many Adam steps
 * 3. planted contacts are recovered from a synthetic alignment
 * 4. MSA parsing: digit mode, FASTA, A3M inserts, filters, column mapping
 * 5. the regressions found in the study are actually fixed
 */

import assert from 'node:assert/strict';
import { naiveGrads } from './naive.mjs';

const core = (await import('../gremlin-core.js')).default;
const MSA = (await import('../msa.js')).default;
const { Gremlin } = core;

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log('  ok   ' + name); passed++; }
  catch (e) { console.log('  FAIL ' + name + '\n       ' + (e.message || e).split('\n').join('\n       ')); failed++; }
}
function section(s) { console.log('\n' + s); }

function lcg(seed) {
  let st = seed >>> 0;
  return () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/* ------------------------------------------------------------------ */
section('1. gradient equivalence with the original implementation');

test('data gradient, bias gradient and PLL match naiveGrads', () => {
  const L = 6, A = 5, N = 7, LA = L * A;
  const rnd = lcg(42);

  const rows = [], seqs = new Int32Array(N * L);
  for (let n = 0; n < N; n++) {
    const row = [];
    for (let i = 0; i < L; i++) { const v = (rnd() * A) | 0; row.push(v); seqs[n * L + i] = v; }
    rows.push(row);
  }

  // uniform sequence weights so coef[n] === 1/N, matching the original's /numSamples
  const sw = new Float32Array(N).fill(1);
  const g = new Gremlin({ L, A, N, seqs, sw, biasInit: 'zero',
                          cfg: { batch: N, alpha: 0, beta: 0, lr: 0 } });

  // one random symmetric coupling tensor, written into both representations
  const Wnaive = Array(LA * LA).fill(0);
  const bias = Array(LA).fill(0).map(() => (rnd() - 0.5) * 0.4);
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      for (let a = 0; a < A; a++) {
        for (let b = 0; b < A; b++) {
          const c = (rnd() - 0.5) * 0.6;
          Wnaive[(i * A + a) * LA + (j * A + b)] = c;
          Wnaive[(j * A + b) * LA + (i * A + a)] = c;
          g.W[((i * L + j) * A + b) * A + a] = c;
          g.W[((j * L + i) * A + a) * A + b] = c;
        }
      }
    }
  }
  for (let k = 0; k < LA; k++) g.b[k] = bias[k];

  const nv = naiveGrads(Wnaive, bias, rows, L, A);
  g.step();                                    // lr = 0, so W is untouched; G holds the gradient

  let maxG = 0;
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      for (let a = 0; a < A; a++) {
        for (let b = 0; b < A; b++) {
          maxG = Math.max(maxG, Math.abs(
            nv.weightGrads[(i * A + a) * LA + (j * A + b)] - g.G[((i * L + j) * A + b) * A + a]));
        }
      }
    }
  }
  let maxB = 0;
  for (let k = 0; k < LA; k++) maxB = Math.max(maxB, Math.abs(nv.biasGrads[k] - g.Gb[k]));

  console.log('       max |dW diff| = ' + maxG.toExponential(2)
            + ', max |db diff| = ' + maxB.toExponential(2)
            + ', |dloss| = ' + Math.abs(nv.totalLoss - g.pll).toExponential(2));
  assert.ok(maxG < 1e-6, 'coupling gradient diverges: ' + maxG);
  assert.ok(maxB < 1e-6, 'bias gradient diverges: ' + maxB);
  assert.ok(Math.abs(nv.totalLoss - g.pll) < 1e-5, 'PLL diverges');
});

test('gradient still matches when the model must pad and A is large', () => {
  const L = 4, A = 21, N = 5, LA = L * A;
  const rnd = lcg(7);
  const rows = [], seqs = new Int32Array(N * L);
  for (let n = 0; n < N; n++) {
    const row = [];
    for (let i = 0; i < L; i++) { const v = (rnd() * A) | 0; row.push(v); seqs[n * L + i] = v; }
    rows.push(row);
  }
  const sw = new Float32Array(N).fill(1);
  // biasInit 'zero' because the oracle below is built with a zero bias
  const g = new Gremlin({ L, A, N, seqs, sw, biasInit: 'zero',
                          cfg: { batch: N, alpha: 0, beta: 0, lr: 0 } });
  const Wnaive = Array(LA * LA).fill(0);
  const bias = Array(LA).fill(0);
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) {
      const c = (rnd() - 0.5) * 0.5;
      Wnaive[(i * A + a) * LA + (j * A + b)] = c;
      Wnaive[(j * A + b) * LA + (i * A + a)] = c;
      g.W[((i * L + j) * A + b) * A + a] = c;
      g.W[((j * L + i) * A + a) * A + b] = c;
    }
  const nv = naiveGrads(Wnaive, bias, rows, L, A);
  g.step();
  let maxG = 0;
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++)
      maxG = Math.max(maxG, Math.abs(
        nv.weightGrads[(i * A + a) * LA + (j * A + b)] - g.G[((i * L + j) * A + b) * A + a]));
  assert.ok(maxG < 1e-6, 'diverges at A=21: ' + maxG);
});

/* ------------------------------------------------------------------ */
section('2. invariants');

test('W stays bit-exactly symmetric over 200 Adam steps', () => {
  const sim = MSA.synthetic({ L: 12, N: 60, A: 6, nPairs: 3, seed: 11 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, cfg: { batch: ds.N, lr: 0.1 } });
  for (let s = 0; s < 200; s++) g.step();
  const { L, A, AA, W } = g;
  let bad = 0;
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      for (let a = 0; a < A; a++) {
        for (let b = 0; b < A; b++) {
          if (W[((i * L + j) * A + b) * A + a] !== W[((j * L + i) * A + a) * A + b]) bad++;
        }
      }
    }
  }
  assert.equal(bad, 0, bad + ' asymmetric entries');
});

test('diagonal blocks stay zero and nothing goes non-finite', () => {
  const sim = MSA.synthetic({ L: 10, N: 50, A: 8, nPairs: 2, seed: 3 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, cfg: { batch: 16, lr: 0.1 } });
  for (let s = 0; s < 150; s++) g.step();
  for (let i = 0; i < g.L; i++) {
    const o = (i * g.L + i) * g.AA;
    for (let k = 0; k < g.AA; k++) assert.equal(g.W[o + k], 0, 'diagonal block written at i=' + i);
  }
  for (let k = 0; k < g.P; k++) assert.ok(Number.isFinite(g.W[k]), 'non-finite coupling');
  assert.ok(Number.isFinite(g.loss), 'non-finite loss');
});

test('softmax survives couplings large enough to overflow the original', () => {
  // the original did arr.map(Math.exp) with no max subtraction: exp(800) = Infinity
  const L = 3, A = 4, N = 2;
  const seqs = Int32Array.from([0, 1, 2, 1, 2, 3]);
  const sw = new Float32Array(N).fill(1);
  const g = new Gremlin({ L, A, N, seqs, sw, biasInit: 'zero',
                          cfg: { batch: N, alpha: 0, beta: 0, lr: 0 } });
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    if (i === j) continue;
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) g.W[((i * L + j) * A + b) * A + a] = 500;
  }
  g.step();
  assert.ok(Number.isFinite(g.pll), 'PLL went non-finite: ' + g.pll);
  for (let k = 0; k < g.G.length; k++) assert.ok(Number.isFinite(g.G[k]), 'gradient went non-finite');
});

test('loss history is bounded and keeps its full span', () => {
  const sim = MSA.synthetic({ L: 8, N: 40, A: 5, nPairs: 2, seed: 5 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, cfg: { batch: ds.N, lr: 0.1 } });
  for (let s = 0; s < 20000; s++) g.step();
  const h = g.history();
  assert.ok(h.length / 2 <= 4096, 'history exceeded its cap: ' + h.length / 2);
  assert.ok(h[h.length - 2] >= 19000, 'history lost the recent end: ' + h[h.length - 2]);
  assert.equal(h[0], 0, 'history lost the origin');
});

/* ------------------------------------------------------------------ */
section('3. contact recovery on planted couplings');

// Planted pairs are indices into the input alignment; the model works in
// post-filter columns. Push them through colMap so the test checks the mapping
// instead of assuming it is the identity.
function plantedInModelSpace(sim, ds) {
  const inv = new Map();
  for (let c = 0; c < ds.L; c++) inv.set(ds.colMap[c], c);
  const out = new Set();
  for (const [i, j] of sim.pairs) {
    if (inv.has(i) && inv.has(j)) {
      const a = inv.get(i), b = inv.get(j);
      out.add(Math.min(a, b) + ',' + Math.max(a, b));
    }
  }
  return out;
}

test('top-scoring pairs are the planted ones (L=48, N=800, A=21)', () => {
  const sim = MSA.synthetic({ L: 48, N: 800, A: 20, nPairs: 8, seed: 20260804 });
  const ds = MSA.buildDataset(sim.text, {});
  assert.equal(ds.mode, 'protein');
  assert.equal(ds.L, 48);
  assert.equal(ds.A, 21);

  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs,
                          cfg: { batch: 256, lr: 0.05, alpha: 0.01, beta: 0.01 }, seed: 1 });
  const t0 = Date.now();
  for (let s = 0; s < 400; s++) g.step();
  const secs = (Date.now() - t0) / 1000;

  const cm = g.contactMap();
  const top = g.topContacts(4, 8, cm);
  const planted = plantedInModelSpace(sim, ds);
  const found = top.filter(c => planted.has(c.i + ',' + c.j)).length;

  console.log('       Meff=' + g.Meff.toFixed(0) + '  ' + secs.toFixed(1) + 's for 400 steps'
            + '  recovered ' + found + '/' + planted.size + ' in the top 8');
  console.log('       top 8: ' + top.map(c => c.i + '-' + c.j + '(' + c.score.toFixed(2) + ')').join(' '));
  assert.ok(found >= 7, 'only recovered ' + found + '/' + planted.size);
});

test('minibatching does not break recovery (B=64 on N=1500)', () => {
  const sim = MSA.synthetic({ L: 32, N: 1500, A: 20, nPairs: 5, seed: 99 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs,
                          cfg: { batch: 64, lr: 0.05 }, seed: 2 });
  for (let s = 0; s < 600; s++) g.step();
  const top = g.topContacts(4, 5);
  const planted = plantedInModelSpace(sim, ds);
  const found = top.filter(c => planted.has(c.i + ',' + c.j)).length;
  console.log('       recovered ' + found + '/' + planted.size + ' with B=64');
  assert.ok(found >= 4, 'only recovered ' + found + '/' + planted.size);
});

test('contact scores survive the gauge: constant shift per block changes nothing', () => {
  const sim = MSA.synthetic({ L: 10, N: 100, A: 6, nPairs: 3, seed: 17 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, cfg: { batch: ds.N, lr: 0.1 } });
  for (let s = 0; s < 100; s++) g.step();
  const before = g.contactMap();
  // add a row-constant to every block: pure gauge, must not move the scores
  const { L, A, AA, W } = g;
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    if (i === j) continue;
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) W[((i * L + j) * A + b) * A + a] += 0.25 * (a + 1);
  }
  const after = g.contactMap();
  let maxd = 0;
  for (let k = 0; k < before.length; k++) maxd = Math.max(maxd, Math.abs(before[k] - after[k]));
  console.log('       max score shift under a gauge transform: ' + maxd.toExponential(2));
  assert.ok(maxd < 1e-3, 'gauge fixing is not working: ' + maxd);
});

/* ------------------------------------------------------------------ */
section('4. MSA input');

test('digit toy format still parses exactly as before', () => {
  const ds = MSA.buildDataset('012\n102\n012\n102', {});
  assert.equal(ds.mode, 'digit');
  assert.equal(ds.L, 3);
  assert.equal(ds.A, 3);
  assert.equal(ds.N, 4);
  assert.deepEqual(Array.from(ds.seqs), [0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 2]);
});

test('protein FASTA reaches A=21, which the original could not represent', () => {
  const ds = MSA.buildDataset('>a\nARNDCQEGHILKMFPSTWYV-\n>b\nARNDCQEGHILKMFPSTWYVA\n', { keepQueryColumns: false });
  assert.equal(ds.mode, 'protein');
  assert.equal(ds.A, 21);
  assert.equal(ds.L, 21);
  assert.deepEqual(Array.from(ds.seqs.slice(0, 21)), [...Array(21).keys()]);
});

test('A3M lowercase insertions are stripped and lengths reconciled', () => {
  const ds = MSA.buildDataset('>q\nACDEF\n>h\nACdefDEF\n', { keepQueryColumns: false, minCoverage: 0 });
  assert.equal(ds.L, 5);
  assert.equal(ds.N, 2);
  assert.ok(ds.warnings.some(w => /A3M insertions/.test(w)), 'no insertion warning: ' + ds.warnings);
});

test('unknown residues fold into the gap state instead of becoming NaN', () => {
  // the original did Number('X') -> NaN -> an all-zero one-hot that silently
  // contributed nothing to any logit
  const ds = MSA.buildDataset('>a\nAXBZ\n', { keepQueryColumns: false, minCoverage: 0 });
  assert.deepEqual(Array.from(ds.seqs), [0, MSA.GAP, MSA.GAP, MSA.GAP]);
});

test('query-gap columns are dropped and colMap preserves query numbering', () => {
  const ds = MSA.buildDataset('>q\nA-C-E\n>h\nAKCKE\n', { minCoverage: 0 });
  assert.equal(ds.L, 3);
  assert.deepEqual(Array.from(ds.colMap), [0, 2, 4]);
});

test('low-coverage sequences are dropped but the query never is', () => {
  const ds = MSA.buildDataset('>q\nACDEFGHIKL\n>ok\nACDEFGHIKL\n>bad\nAC--------\n',
                              { minCoverage: 0.5 });
  assert.equal(ds.N, 2);
  assert.ok(ds.warnings.some(w => /below 0.50 coverage/.test(w)), ds.warnings.join('; '));
});

/* ---- py2Dmol-compatible filtering, so the practical page matches it ---- */

test('coverage and identity match py2Dmol definitions (X counts as a gap)', () => {
  assert.equal(MSA.coverageOf('AC--'), 0.5);
  assert.equal(MSA.coverageOf('ACXY'), 0.75);          // X is a gap for coverage
  assert.equal(MSA.identityTo('ACDE', 'ACDE'), 1);
  assert.equal(MSA.identityTo('ACDE', 'ACDF'), 0.75);
  assert.equal(MSA.identityTo('AXDE', 'AXDE'), 0.75);  // X never counts as a match
  assert.equal(MSA.identityTo('----', 'ACDE'), 0);
});

test('identity filter keeps the query and drops distant sequences', () => {
  const text = '>q\nACDEFGHIKL\n>near\nACDEFGHIKW\n>far\nWWWWWWWWWW\n';
  const ds = MSA.buildDataset(text, { keepQueryColumns: false, minIdentity: 0.5 });
  assert.equal(ds.N, 2);
  assert.ok(ds.warnings.some(w => /identity to the query/.test(w)), ds.warnings.join('; '));
});

test('sortByIdentity orders by descending identity with the query first', () => {
  const text = '>q\nACDEFGHIKL\n>mid\nACDEFGHWWW\n>high\nACDEFGHIKW\n';
  const ds = MSA.buildDataset(text, { keepQueryColumns: false, sortByIdentity: true });
  assert.deepEqual(ds.names, ['q', 'high', 'mid']);
  assert.ok(ds.idn[1] > ds.idn[2], 'identities not descending: ' + Array.from(ds.idn));
  assert.equal(ds.idn[0], 1);
});

test('afdbMsaUrl builds the path py2Dmol uses', () => {
  assert.equal(MSA.afdbMsaUrl(' p0a7y4 '),
    'https://alphafold.ebi.ac.uk/files/msa/AF-P0A7Y4-F1-msa_v6.a3m');
});

/* ---- educational page must stay numerically faithful to the original ---- */

test("regMode 'raw' + uniformWeights reproduce the original's penalty exactly", () => {
  // original: weightGrads[i] += alpha * w[i];  biasGrads[i] += beta * b[i]
  const L = 4, A = 3, N = 3, alpha = 0.1, beta = 0.05;
  const seqs = Int32Array.from([0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 2]);
  const g = new Gremlin({
    L, A, N, seqs, uniformWeights: true,
    cfg: { batch: N, alpha, beta, lr: 0, regMode: 'raw' }
  });
  assert.equal(g.Meff, N, 'uniformWeights should give Meff = N');

  // a known W/b, then compare the L2 part of the gradient against alpha*w
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) {
      g.W[((i * L + j) * A + b) * A + a] = 0.3;
      g.W[((j * L + i) * A + a) * A + b] = 0.3;
    }
  for (let k = 0; k < L * A; k++) g.b[k] = 0.2;

  // capture the data-only gradient (alpha=beta=0), then the regularized one
  const plain = new Gremlin({ L, A, N, seqs, uniformWeights: true,
    cfg: { batch: N, alpha: 0, beta: 0, lr: 0, regMode: 'raw' } });
  plain.W.set(g.W); plain.b.set(g.b);
  plain.step();
  g.step();

  // Adam normalizes the step itself, so check the penalty the objective reports:
  // raw mode must give (alpha/2)*sum(w^2), matching the original's 0.5*alpha*sum.
  // Non-diagonal ordered blocks: L*(L-1) of them, A*A entries each.
  const sw2 = L * (L - 1) * A * A * 0.3 * 0.3;
  assert.ok(Math.abs(g.regW - (alpha / 2) * sw2) < 1e-3,
    'raw regW ' + g.regW + ' != ' + (alpha / 2) * sw2);
  const sb2 = L * A * 0.2 * 0.2;
  assert.ok(Math.abs(g.regB - (beta / 2) * sb2) < 1e-4,
    'raw regB ' + g.regB + ' != ' + (beta / 2) * sb2);
  // and the data gradient must be untouched by the mode
  assert.ok(Math.abs(plain.pll - g.pll) < 1e-6, 'pll should not depend on regMode');
});

test("regMode 'gremlin' scales the penalty by (L-1)(A-1)/Meff", () => {
  const L = 4, A = 3, N = 3, alpha = 0.1;
  const seqs = Int32Array.from([0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 2]);
  const mk = (regMode) => {
    const g = new Gremlin({ L, A, N, seqs, uniformWeights: true,
      cfg: { batch: N, alpha, beta: 0, lr: 0, regMode } });
    for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
      for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) {
        g.W[((i * L + j) * A + b) * A + a] = 0.3;
        g.W[((j * L + i) * A + a) * A + b] = 0.3;
      }
    g.step();
    return g.regW;
  };
  // raw: lam = alpha/2.  gremlin: lam = 0.5*alpha*(L-1)*(A-1)/Meff, the 0.5
  // matching the reference's 0.5*(L-1)*(A-1)*sum(w^2).  ratio = (L-1)(A-1)/Meff.
  const ratio = mk('gremlin') / mk('raw');
  const want = (L - 1) * (A - 1) / N;
  assert.ok(Math.abs(ratio - want) < 1e-3, 'ratio ' + ratio + ' != ' + want);
});

test('column gap filter works and reports what it kept', () => {
  const ds = MSA.buildDataset('>q\nACDEF\n>b\nA-DEF\n>c\nA-DEF\n>d\nA-DEF\n',
                              { maxColGap: 0.5, minCoverage: 0 });
  assert.equal(ds.L, 4);
  assert.deepEqual(Array.from(ds.colMap), [0, 2, 3, 4]);
});

test('ragged input is padded rather than crashing', () => {
  const ds = MSA.buildDataset('>a\nACDEF\n>b\nACD\n', { keepQueryColumns: false, minCoverage: 0 });
  assert.equal(ds.L, 5);
  assert.ok(ds.warnings.some(w => /padded/.test(w)));
});

test('100k sequences parse without a spread-argument stack overflow', () => {
  // the original used Math.max(...) over N and over N*L values; a 1000x100
  // alignment already exceeds the argument limit
  const rows = [];
  for (let n = 0; n < 100000; n++) rows.push('ACDE');
  const ds = MSA.buildDataset(rows.join('\n'), { keepQueryColumns: false, minCoverage: 0 });
  assert.equal(ds.N, 100000);
  assert.equal(ds.L, 4);
});

test('the redundancy filter drops near-duplicates and keeps the query', () => {
  const text = '>q\nACDEFGHIKLACDEFGHIKL\n'
             + '>dup\nACDEFGHIKLACDEFGHIKL\n'      // identical to the query
             + '>near\nACDEFGHIKLACDEFGHIKW\n'     // 19/20 = 0.95
             + '>far\nWWWWWWWWWWWWWWWWWWWW\n';
  const off = MSA.buildDataset(text, { keepQueryColumns: false });
  assert.equal(off.N, 4);

  const at90 = MSA.buildDataset(text, { keepQueryColumns: false, maxIdentity: 0.9 });
  assert.deepEqual(at90.names, ['q', 'far'], JSON.stringify(at90.names));
  assert.equal(at90.redundancy.dropped, 2);

  // 0.95 is not "more than 0.95", so `near` survives a 0.96 threshold
  const at96 = MSA.buildDataset(text, { keepQueryColumns: false, maxIdentity: 0.96 });
  assert.deepEqual(at96.names, ['q', 'near', 'far'], JSON.stringify(at96.names));
});

test('the redundancy filter keeps seqs, names, cov and idn in step', () => {
  const sim = MSA.synthetic({ L: 20, N: 200, A: 6, nPairs: 2, seed: 64 });
  const ds = MSA.buildDataset(sim.text, { maxIdentity: 0.8 });
  assert.equal(ds.names.length, ds.N, 'names out of step');
  assert.equal(ds.cov.length, ds.N, 'cov out of step');
  assert.equal(ds.idn.length, ds.N, 'idn out of step');
  assert.equal(ds.seqs.length, ds.N * ds.L, 'seqs out of step');
  for (let k = 0; k < ds.seqs.length; k++) {
    assert.ok(ds.seqs[k] >= 0 && ds.seqs[k] < ds.A, 'bad state after filtering');
  }
  console.log('       ' + sim.N + ' -> ' + ds.N + ' sequences at id <= 0.80');
  assert.ok(ds.N < sim.N, 'nothing was filtered on a redundant synthetic set');
});

test('no pair above the threshold survives the filter', () => {
  const sim = MSA.synthetic({ L: 24, N: 150, A: 5, nPairs: 2, seed: 8 });
  const ds = MSA.buildDataset(sim.text, { maxIdentity: 0.75 });
  const need = Math.ceil(0.75 * ds.L);
  let worst = 0;
  for (let n = 0; n < ds.N; n++) {
    for (let m = n + 1; m < ds.N; m++) {
      let id = 0;
      for (let k = 0; k < ds.L; k++) if (ds.seqs[n * ds.L + k] === ds.seqs[m * ds.L + k]) id++;
      if (id > worst) worst = id;
    }
  }
  console.log('       highest surviving pair identity: ' + (worst / ds.L).toFixed(3)
            + ' (threshold ' + (need / ds.L).toFixed(3) + ')');
  assert.ok(worst < need, 'a pair above the threshold survived: ' + worst + ' >= ' + need);
});

/*
 * An alignment with redundancy you can count: `founders` unrelated sequences,
 * each copied `perFounder` times with `mut` positions randomized. The synthetic
 * generator makes sequences that are all mutually distant, which leaves the
 * clustering tests vacuous (400 sequences, 400 clusters).
 */
function clustered({ founders, perFounder, L, mut, seed }) {
  let s = seed >>> 0;
  const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const AB = 'ACDEFGHIKLMNPQRSTVWY';
  const rows = [];
  for (let f = 0; f < founders; f++) {
    const base = [];
    for (let k = 0; k < L; k++) base.push(AB[(rnd() * AB.length) | 0]);
    for (let c = 0; c < perFounder; c++) {
      const v = base.slice();
      for (let m = 0; m < mut; m++) v[(rnd() * L) | 0] = AB[(rnd() * AB.length) | 0];
      rows.push('>f' + f + '_' + c + '\n' + v.join(''));
    }
  }
  return rows.join('\n') + '\n';
}

test('cluster weights and the filter compute the same partition', () => {
  // Meff under cluster weighting is the number of clusters, and the filter
  // keeps exactly one sequence per cluster -- so the two must agree on the
  // count even though one discards data and the other does not.
  const text = clustered({ founders: 12, perFounder: 20, L: 60, mut: 3, seed: 21 });
  const full = MSA.buildDataset(text, {});
  const filtered = MSA.buildDataset(text, { maxIdentity: 0.8 });
  assert.ok(filtered.N < full.N / 2, 'the fixture is not actually redundant');

  const g = new Gremlin({
    L: full.L, A: full.A, N: full.N, seqs: full.seqs,
    identity: 0.8, weightMode: 'cluster', seed: 1
  });
  assert.equal(g.weightMode, 'cluster');
  assert.equal(g.clusters, filtered.N,
    'cluster count ' + g.clusters + ' != kept sequences ' + filtered.N);
  // Meff = sum of 1/|cluster| = number of clusters, up to float32 accumulation
  assert.ok(Math.abs(g.Meff - g.clusters) < 0.01 * g.clusters,
    'Meff ' + g.Meff.toFixed(2) + ' != cluster count ' + g.clusters);
  console.log('       N ' + full.N + ' -> ' + g.clusters + ' clusters; '
            + 'filter kept ' + filtered.N + ', cluster Meff ' + g.Meff.toFixed(1));
});

test('cluster weights approximate exact Meff from above', () => {
  // Mutate hard enough that identities straddle the threshold, so the identity
  // graph is NOT a disjoint union of cliques and the hard partition has to
  // split neighbourhoods the exact count treats as overlapping. That is the
  // case where the two disagree, and the direction of the disagreement is the
  // thing worth pinning down.
  const mk = (text, weightMode) => {
    const ds = MSA.buildDataset(text, {});
    return new Gremlin({
      L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, identity: 0.8, weightMode, seed: 1
    });
  };
  let sawGap = false;
  for (const mut of [8, 11, 14]) {
    const text = clustered({ founders: 8, perFounder: 25, L: 60, mut, seed: 7 });
    const exact = mk(text, 'exact'), cluster = mk(text, 'cluster');
    console.log('       ' + mut + ' mutations: exact Meff ' + exact.Meff.toFixed(1)
              + ', cluster Meff ' + cluster.Meff.toFixed(1)
              + ' (' + cluster.clusters + ' clusters)');
    assert.ok(cluster.Meff >= exact.Meff - 1e-3,
      'cluster Meff ' + cluster.Meff + ' below exact ' + exact.Meff);
    assert.ok(cluster.Meff < 3 * exact.Meff, 'cluster Meff wildly off');
    assert.equal(exact.approxWeights, false);
    assert.equal(cluster.approxWeights, true);
    if (cluster.Meff > exact.Meff + 1e-3) sawGap = true;
  }
  assert.ok(sawGap, 'never hit a case where the partition and the pair count differ');
});

test('reweighting is skipped when the filter already ran at that threshold', () => {
  const sim = MSA.synthetic({ L: 24, N: 300, A: 6, nPairs: 2, seed: 5 });
  const ds = MSA.buildDataset(sim.text, { maxIdentity: 0.8 });
  const mk = (opts) => new Gremlin(Object.assign({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, identity: 0.8, seed: 1
  }, opts));

  // no pair survives above 0.8, so the O(N^2) pass provably returns Meff = N
  const done = mk({ filteredAt: 0.8 });
  assert.equal(done.weightMode, 'filtered');
  assert.equal(done.Meff, ds.N);
  assert.equal(done.approxWeights, false, 'the skip is exact, not an approximation');

  // and the skip agrees with actually running it
  const ran = mk({ weightMode: 'exact' });
  assert.ok(Math.abs(ran.Meff - done.Meff) < 1e-3,
    'skip claimed ' + done.Meff + ' but the exact pass found ' + ran.Meff);

  // below the filter threshold the skip must NOT fire: pairs in [Tm, Tf) count
  const lower = mk({ filteredAt: 0.8, identity: 0.5, weightMode: 'exact' });
  assert.notEqual(lower.weightMode, 'filtered');
  assert.ok(lower.Meff < ds.N, 'reweighting at 0.5 found no neighbours at all');
  console.log('       filtered at 0.8: Meff ' + done.Meff + ' (skipped) == ' + ran.Meff.toFixed(1)
            + ' (computed); at threshold 0.5 it still finds ' + lower.Meff.toFixed(1));
});

/* ------------------------------------------------------------------ */
section('5. cost model');

test('parameter memory matches the O(L^2 A^2) projection', () => {
  assert.equal(Gremlin.bytes(64, 21), 4 * 4 * 64 * 64 * 441);
  const mb = (L) => Gremlin.bytes(L, 21) / 1048576;
  console.log('       W+G+m+v:  L=64 ' + mb(64).toFixed(0) + 'MB   L=128 ' + mb(128).toFixed(0)
            + 'MB   L=256 ' + mb(256).toFixed(0) + 'MB   L=384 ' + mb(384).toFixed(0) + 'MB');
  assert.ok(mb(256) > 400 && mb(256) < 500);
});

test('a step costs O(L^2 A B), independent of N', () => {
  const mk = (N) => {
    const sim = MSA.synthetic({ L: 24, N, A: 21, nPairs: 4, seed: 8 });
    const ds = MSA.buildDataset(sim.text, {});
    return new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, cfg: { batch: 128, lr: 0.05 } });
  };
  const timeIt = (g) => {
    for (let s = 0; s < 5; s++) g.step();                 // warm up
    const t0 = process.hrtime.bigint();
    for (let s = 0; s < 30; s++) g.step();
    return Number(process.hrtime.bigint() - t0) / 1e6 / 30;
  };
  const small = timeIt(mk(400));
  const big = timeIt(mk(4000));
  console.log('       ms/step at N=400: ' + small.toFixed(2) + '   at N=4000: ' + big.toFixed(2)
            + '   ratio ' + (big / small).toFixed(2));
  assert.ok(big < small * 2.5, '10x more sequences should not cost 2.5x more per step');
});

/* ------------------------------------------------------------------ */
console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
