/*
 * node test/backends.test.mjs
 *
 * The WASM backend, and the self-test gate that decides whether a backend is
 * allowed to run at all.
 *
 * The gate is the reason the WebGPU path can be shipped from an environment
 * where WebGPU cannot be executed: a backend that computes the wrong gradient
 * refuses itself and the caller falls back. These tests check that it really
 * does reject a wrong backend, not just accept a right one -- a gate that never
 * fires is worse than no gate, because it looks like safety.
 */

import assert from 'node:assert/strict';

const core = (await import('../gremlin-core.js')).default;
const MSA = (await import('../msa.js')).default;
const { Gremlin } = core;
const WASM_URL = new URL('../gremlin.wasm', import.meta.url).pathname;

let passed = 0, failed = 0;
async function test(name, fn) {
  try { await fn(); console.log('  ok   ' + name); passed++; }
  catch (e) { console.log('  FAIL ' + name + '\n       ' + String(e.message || e).split('\n').join('\n       ')); failed++; }
}
const section = (s) => console.log('\n' + s);

const wasm = await core.initWasm(WASM_URL);
if (!wasm) {
  console.log('WASM failed to load: ' + core.lastWasmError);
  console.log('run ./build-wasm.sh first');
  process.exit(1);
}

/* ------------------------------------------------------------------ */
section('1. exp / log approximations');

await test('fast_expf is within 1e-5 relative over the range softmax uses', () => {
  let worst = 0, at = 0;
  for (let x = -87; x <= 0; x += 0.0005) {
    const g = wasm.x.probe_exp(x), e = Math.exp(x);
    const d = Math.abs(g - e) / e;
    if (d > worst) { worst = d; at = x; }
  }
  console.log('       max rel err ' + worst.toExponential(2) + ' at x=' + at.toFixed(3)
            + ' (f32 eps ~1.2e-7)');
  assert.ok(worst < 1e-5, 'exp too inaccurate: ' + worst);
});

await test('fast_expf underflows to zero only below the f32 subnormal range', () => {
  assert.equal(wasm.x.probe_exp(-87.5), 0);
  assert.ok(wasm.x.probe_exp(-87) > 0, 'cut off too early');
  assert.equal(wasm.x.probe_exp(0), 1, 'exp(0) must be exactly 1');
});

await test('fast_logf is within 1e-5 absolute over [1, A]', () => {
  // only ever called on a softmax denominator, which is in [1, A]
  let worst = 0;
  for (let x = 1; x <= 32; x += 0.0002) {
    const d = Math.abs(wasm.x.probe_log(x) - Math.log(x));
    if (d > worst) worst = d;
  }
  console.log('       max abs err ' + worst.toExponential(2));
  assert.ok(worst < 1e-5, 'log too inaccurate: ' + worst);
});

/* ------------------------------------------------------------------ */
section('2. the self-test gate');

await test('the real WASM backend passes', async () => {
  const dev = await core.selfTest(wasm);
  console.log('       deviation from JS reference: ' + dev.toExponential(2)
            + '  (tolerance ' + core.SELFTEST_TOL + ')');
  assert.ok(dev <= core.SELFTEST_TOL, 'wasm rejected: ' + dev);
});

/*
 * Sabotage tests. Each wraps the real backend so exactly one kernel returns
 * something wrong, and checks the gate notices. If any of these pass the gate,
 * the safety argument for shipping the unverified GPU path collapses.
 */
function sabotage(kernel, mutate) {
  const fake = Object.create(Object.getPrototypeOf(wasm));
  Object.assign(fake, wasm);
  fake.x = Object.assign({}, wasm.x);
  fake.x[kernel] = function () {
    const r = wasm.x[kernel].apply(null, arguments);
    mutate.apply(null, arguments);
    return r;
  };
  return fake;
}

await test('a corrupted coupling gradient is rejected', async () => {
  // scale G by 1.05 after the data pass
  const bad = sabotage('data_pass', function (Wp, Gp) {
    const v = new Float32Array(wasm.memory.buffer, Gp, 64);
    for (let k = 0; k < v.length; k++) v[k] *= 1.05;
  });
  const dev = await core.selfTest(bad);
  console.log('       deviation ' + dev.toExponential(2) + ' -> ' + (dev > core.SELFTEST_TOL ? 'rejected' : 'ACCEPTED'));
  assert.ok(dev > core.SELFTEST_TOL, 'gate accepted a corrupted gradient (' + dev + ')');
});

await test('a skipped symmetrization is rejected', async () => {
  const fake = Object.create(Object.getPrototypeOf(wasm));
  Object.assign(fake, wasm);
  fake.x = Object.assign({}, wasm.x, { symmetrize: function () { /* no-op */ } });
  const dev = await core.selfTest(fake);
  console.log('       deviation ' + dev.toExponential(2) + ' -> ' + (dev > core.SELFTEST_TOL ? 'rejected' : 'ACCEPTED'));
  assert.ok(dev > core.SELFTEST_TOL, 'gate accepted an unsymmetrized gradient');
});

await test('a wrong Adam update is rejected', async () => {
  const fake = Object.create(Object.getPrototypeOf(wasm));
  Object.assign(fake, wasm);
  fake.x = Object.assign({}, wasm.x, {
    adam: function (Wp, Gp, Mp, Vp, P, lr) {
      // right shape, wrong learning rate -- the kind of bug a bad uniform gives
      return wasm.x.adam(Wp, Gp, Mp, Vp, P, lr * 1.5,
                         arguments[6], arguments[7], arguments[8],
                         arguments[9], arguments[10], arguments[11]);
    }
  });
  const dev = await core.selfTest(fake);
  console.log('       deviation ' + dev.toExponential(2) + ' -> ' + (dev > core.SELFTEST_TOL ? 'rejected' : 'ACCEPTED'));
  assert.ok(dev > core.SELFTEST_TOL, 'gate accepted a wrong Adam step');
});

await test('a wrong contact map is rejected', async () => {
  const fake = Object.create(Object.getPrototypeOf(wasm));
  Object.assign(fake, wasm);
  fake.x = Object.assign({}, wasm.x, {
    contact_map: function (Wp, outp, scratchp, L, A, nA) {
      wasm.x.contact_map(Wp, outp, scratchp, L, A, nA);   // full signature: the
      // sabotage must be the offset below, not a dropped argument
      const v = new Float32Array(wasm.memory.buffer, outp, L * L);
      for (let i = 0; i < L * L; i++) v[i] += 0.05;       // plausible, wrong
    }
  });
  const dev = await core.selfTest(fake);
  console.log('       deviation ' + dev.toExponential(2) + ' -> ' + (dev > core.SELFTEST_TOL ? 'rejected' : 'ACCEPTED'));
  assert.ok(dev > core.SELFTEST_TOL, 'gate accepted a wrong contact map');
});

await test('selectBackend falls back to JS when WASM cannot load', async () => {
  const sel = await core.selectBackend({ wasmUrl: '/nonexistent/gremlin.wasm' });
  assert.equal(sel.name, 'js', 'should have fallen back, got ' + sel.name);
  assert.ok(sel.tried.some(t => t.name === 'wasm' && !t.ok), JSON.stringify(sel.tried));
  console.log('       tried: ' + sel.tried.map(t => t.name + '=' + (t.ok ? 'ok' : 'no')).join(' '));
});

await test('selectBackend picks WASM when it is available', async () => {
  const sel = await core.selectBackend({ wasmUrl: WASM_URL });
  assert.equal(sel.name, 'wasm');
  const w = sel.tried.find(t => t.name === 'wasm');
  console.log('       chose ' + sel.name + ', deviation ' + w.dev.toExponential(2));
});

/* ------------------------------------------------------------------ */
section('3. WASM matches JS on a realistic problem');

await test('a single step agrees tightly from identical parameters', async () => {
  // This is the real numerical claim: given the same W and b, both backends
  // compute the same gradient and the same updated parameters.
  const sim = MSA.synthetic({ L: 24, N: 500, A: 20, nPairs: 4, seed: 4242 });
  const ds = MSA.buildDataset(sim.text, {});
  const mk = (backend) => new Gremlin({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, uniformWeights: true, backend,
    cfg: { batch: ds.N, lr: 0.05, alpha: 0.01, beta: 0.01 }, seed: 7
  });
  const a = mk(null), b = mk(wasm);
  // a non-trivial, symmetric starting point
  let st = 31337;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const { L, A } = a;
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      for (let p = 0; p < A; p++) {
        for (let q = 0; q < A; q++) {
          const c = (rnd() - 0.5) * 0.4;
          a.W[((i * L + j) * A + q) * A + p] = c;
          a.W[((j * L + i) * A + p) * A + q] = c;
        }
      }
    }
  }
  for (let k = 0; k < L * A; k++) a.b[k] = (rnd() - 0.5) * 0.3;
  b.W.set(a.W); b.b.set(a.b);

  a.step(); b.step();
  let gDev = 0, wDev = 0;
  for (let k = 0; k < a.P; k++) {
    gDev = Math.max(gDev, Math.abs(a.G[k] - b.G[k]) / (Math.abs(a.G[k]) + 1e-4));
    wDev = Math.max(wDev, Math.abs(a.W[k] - b.W[k]) / (Math.abs(a.W[k]) + 1e-4));
  }
  const lDev = Math.abs(a.pll - b.pll) / Math.abs(a.pll);
  console.log('       one step: max dG ' + gDev.toExponential(2)
            + ', max dW ' + wDev.toExponential(2) + ', dPLL ' + lDev.toExponential(2));
  assert.ok(gDev < 1e-4, 'gradient disagrees: ' + gDev);
  assert.ok(lDev < 1e-5, 'PLL disagrees: ' + lDev);
  /*
   * W is held to a looser bound than G on purpose. Adam's first step is
   * lr * g / (|g| + eps) ~ lr * sign(g), which is discontinuous at g = 0, so
   * wherever the gradient is near zero a 7e-6 difference can move a parameter by
   * a full lr. That is a property of Adam, not of the backend -- hence the
   * gradient check above is the one with teeth.
   */
  assert.ok(wDev < 2e-2, 'update disagrees far more than Adam sign-flips explain: ' + wDev);
});

await test('over 40 steps the loss tracks and the top contacts agree', async () => {
  /*
   * Deliberately NOT comparing parameters element-by-element here. Adam's update
   * is m/sqrt(v), which is scale-free and ill-conditioned wherever the gradient
   * is near zero, so a ~8e-6 difference in exp() is enough to send an individual
   * near-zero coupling one lr-sized step the other way. That shows up as a huge
   * *relative* deviation on a parameter whose true value is ~0 while changing
   * nothing that matters. What has to agree is the loss trajectory and the
   * ranking the contact map produces.
   */
  const sim = MSA.synthetic({ L: 24, N: 500, A: 20, nPairs: 4, seed: 4242 });
  const ds = MSA.buildDataset(sim.text, {});
  const mk = (backend) => new Gremlin({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, uniformWeights: true, backend,
    cfg: { batch: ds.N, lr: 0.05, alpha: 0.01, beta: 0.01 }, seed: 7
  });
  const a = mk(null), b = mk(wasm);
  for (let s = 0; s < 40; s++) { a.step(); b.step(); }

  const lDev = Math.abs(a.loss - b.loss) / Math.abs(a.loss);
  const K = 20;
  const key = (c) => c.i + ',' + c.j;
  const topA = new Set(a.topContacts(4, K).map(key));
  const topB = b.topContacts(4, K).map(key);
  const overlap = topB.filter(k => topA.has(k)).length;
  console.log('       after 40 steps: dLoss ' + lDev.toExponential(2)
            + ', top-' + K + ' overlap ' + overlap + '/' + K);
  assert.ok(lDev < 2e-2, 'loss trajectories diverged: ' + lDev);
  assert.ok(overlap >= K - 2, 'contact ranking diverged: ' + overlap + '/' + K);
});

await test('WASM recovers the same planted contacts as JS', async () => {
  const sim = MSA.synthetic({ L: 32, N: 800, A: 20, nPairs: 5, seed: 555 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, backend: wasm,
    cfg: { batch: 256, lr: 0.05 }, seed: 3
  });
  for (let s = 0; s < 300; s++) g.step();
  const inv = new Map();
  for (let c = 0; c < ds.L; c++) inv.set(ds.colMap[c], c);
  const planted = new Set(sim.pairs
    .filter(([i, j]) => inv.has(i) && inv.has(j))
    .map(([i, j]) => {
      const a = inv.get(i), b = inv.get(j);
      return Math.min(a, b) + ',' + Math.max(a, b);
    }));
  const top = g.topContacts(4, planted.size);
  const found = top.filter(c => planted.has(c.i + ',' + c.j)).length;
  console.log('       recovered ' + found + '/' + planted.size + ' on the WASM backend');
  assert.ok(found >= planted.size - 1, 'only ' + found + '/' + planted.size);
});

await test('W stays exactly symmetric on the WASM backend', async () => {
  const sim = MSA.synthetic({ L: 14, N: 100, A: 8, nPairs: 3, seed: 21 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, backend: wasm,
                          cfg: { batch: ds.N, lr: 0.1 } });
  for (let s = 0; s < 120; s++) g.step();
  const { L, A, W } = g;
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

await test('changing batch size mid-run does not reallocate WASM memory', async () => {
  const sim = MSA.synthetic({ L: 16, N: 600, A: 20, nPairs: 3, seed: 9 });
  const ds = MSA.buildDataset(sim.text, {});
  const g = new Gremlin({ L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, backend: wasm,
                          cfg: { batch: 64, lr: 0.05 } });
  const bytes = wasm.memory.buffer.byteLength;
  const wBuf = g.W.buffer;
  for (let s = 0; s < 5; s++) g.step();
  g.setConfig({ batch: 512 });
  for (let s = 0; s < 5; s++) g.step();
  assert.equal(wasm.memory.buffer.byteLength, bytes, 'memory grew, detaching views');
  assert.ok(g.W.buffer === wBuf, 'W view was replaced');
  assert.ok(Number.isFinite(g.loss), 'loss went non-finite after a batch change');
  console.log('       batch 64 -> 512 with no reallocation, loss ' + g.loss.toFixed(4));
});

/* ------------------------------------------------------------------ */
section('4. reference conventions (sokrypton/laxy gremlin_jax.ipynb)');

await test("regMode 'gremlin' matches the reference's 0.5*(L-1)*(A-1) penalty", () => {
  // reference: l2 = 0.5*(L-1)*(A-1)*sum(w^2) + sum(b^2); loss = cce.sum() + lam*l2
  // ours is that objective divided by Meff, so lam maps to alpha directly.
  const L = 4, A = 3, N = 3, alpha = 0.1;
  const seqs = Int32Array.from([0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 2]);
  const g = new Gremlin({ L, A, N, seqs, uniformWeights: true, gap: -1, biasInit: 'zero',
                          cfg: { batch: N, alpha, beta: 0, lr: 0, regMode: 'gremlin' } });
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) {
      g.W[((i * L + j) * A + b) * A + a] = 0.3;
      g.W[((j * L + i) * A + a) * A + b] = 0.3;
    }
  g.step();
  const sw2 = L * (L - 1) * A * A * 0.09;
  const want = 0.5 * alpha * (L - 1) * (A - 1) / N * sw2;
  assert.ok(Math.abs(g.regW - want) < 1e-4, 'regW ' + g.regW + ' != ' + want);
});

await test('the contact norm excludes the gap state, as the reference does', () => {
  // reference: raw = sqrt(sum(square(W[:,:20,:,20]))) -- "note: we ignore gaps"
  const L = 4, A = 21, N = 3;
  const seqs = new Int32Array(L * N);
  const g = new Gremlin({ L, A, N, seqs, uniformWeights: true, gap: 20, biasInit: 'zero',
                          cfg: { batch: N, lr: 0 } });
  assert.equal(g.normA, 20, 'normA should drop the gap state');
  const before = g.contactMap();
  // put a large coupling in the gap row/column only; scores must not move
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    if (i === j) continue;
    for (let a = 0; a < A; a++) {
      g.W[((i * L + j) * A + 20) * A + a] = 5.0;    // partner state = gap
      g.W[((i * L + j) * A + a) * A + 20] = 5.0;    // own state = gap
    }
  }
  const after = g.contactMap();
  let maxd = 0;
  for (let k = 0; k < before.length; k++) maxd = Math.max(maxd, Math.abs(before[k] - after[k]));
  console.log('       score shift from gap-only couplings: ' + maxd.toExponential(2));
  assert.ok(maxd < 1e-5, 'gap couplings leaked into the contact score: ' + maxd);
});

await test('bias init reproduces the reference formula', () => {
  // b_ini = log(counts + 0.01*log(N)); b = b_ini - mean(b_ini)
  const L = 3, A = 4, N = 5;
  const seqs = Int32Array.from([0,1,2, 0,1,3, 0,2,2, 1,1,2, 0,1,2]);
  const g = new Gremlin({ L, A, N, seqs, uniformWeights: true, gap: -1, biasInit: 'freq',
                          cfg: { batch: N, lr: 0 } });
  const pseudo = 0.01 * Math.log(N);
  for (let i = 0; i < L; i++) {
    const counts = new Array(A).fill(0);
    for (let n = 0; n < N; n++) counts[seqs[n * L + i]]++;
    const raw = counts.map(c => Math.log(c + pseudo));
    const mean = raw.reduce((a, b) => a + b, 0) / A;
    for (let a = 0; a < A; a++) {
      assert.ok(Math.abs(g.b[i * A + a] - (raw[a] - mean)) < 1e-5,
        'bias[' + i + ',' + a + '] = ' + g.b[i * A + a] + ' want ' + (raw[a] - mean));
    }
  }
});

await test("optMode 'gremlin' (scalar second moment) agrees between JS and WASM", async () => {
  // GREMLIN_TF v2.1: vt is a scalar (sum(g*g)) per tensor, no bias correction
  const sim = MSA.synthetic({ L: 18, N: 400, A: 20, nPairs: 3, seed: 77 });
  const ds = MSA.buildDataset(sim.text, {});
  const mk = (backend) => new Gremlin({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, uniformWeights: true, backend,
    gap: 20, biasInit: 'freq',
    cfg: { batch: ds.N, lr: 1.0, alpha: 0.01, beta: 0.01, optMode: 'gremlin' }, seed: 5
  });
  const a = mk(null), b = mk(wasm);
  for (let s = 0; s < 25; s++) { a.step(); b.step(); }
  const lDev = Math.abs(a.loss - b.loss) / Math.abs(a.loss);
  console.log('       after 25 steps: dLoss ' + lDev.toExponential(2)
            + ', scalar vt ' + a.vt.toExponential(3));
  assert.ok(lDev < 1e-3, 'JS and WASM disagree under optMode gremlin: ' + lDev);
  assert.ok(a.vt > 0, 'scalar second moment never accumulated');
  assert.ok(Number.isFinite(a.loss), 'non-finite loss');
});

await test('suggestLr follows 0.1*log(batch)/L', () => {
  assert.ok(Math.abs(Gremlin.suggestLr(155, 128) - 0.1 * Math.log(128) / 155) < 1e-12);
  assert.ok(Gremlin.suggestLr(155, 128) < Gremlin.suggestLr(48, 128), 'should shrink with L');
});

/* ------------------------------------------------------------------ */
section('5. speed');

await test('WASM is faster than JS', async () => {
  const A = 21;
  console.log('    L      N |   B | js ms | wasm ms | speedup');
  for (const [L, N, B] of [[48, 800, 256], [96, 2000, 128], [155, 4000, 128]]) {
    let st = 7;
    const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
    const seqs = new Int32Array(L * N);
    for (let k = 0; k < L * N; k++) seqs[k] = (rnd() * A) | 0;
    const mk = (backend) => new Gremlin({ L, A, N, seqs, uniformWeights: true, backend,
                                          cfg: { batch: B, lr: 0.05 } });
    const time = (g, reps) => {
      for (let s = 0; s < 3; s++) g.step();
      const t0 = process.hrtime.bigint();
      for (let s = 0; s < reps; s++) g.step();
      return Number(process.hrtime.bigint() - t0) / 1e6 / reps;
    };
    const reps = L >= 155 ? 4 : 8;
    const tj = time(mk(null), reps), tw = time(mk(wasm), reps);
    console.log('  ' + String(L).padStart(3) + ' ' + String(N).padStart(6) + ' | '
      + String(B).padStart(3) + ' | ' + tj.toFixed(0).padStart(5) + ' | '
      + tw.toFixed(0).padStart(7) + ' | ' + (tj / tw).toFixed(2) + 'x');
    assert.ok(tw < tj, 'wasm slower than js at L=' + L);
  }
});

console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
