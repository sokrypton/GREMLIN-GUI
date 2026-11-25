/*
 * node test/bench-step.mjs [L] [B] [N]
 *
 * Per-kernel timing of one WASM optimizer step on a random alignment, so a
 * change to gremlin.c can be judged by where the milliseconds actually went.
 */
const core = (await import('../gremlin-core.js')).default;
const { Gremlin } = core;
const wasm = await core.initWasm(new URL('../gremlin.wasm', import.meta.url).pathname);
if (!wasm) { console.log('WASM failed: ' + core.lastWasmError); process.exit(1); }

const L = +(process.argv[2] || 155), B = +(process.argv[3] || 128), N = +(process.argv[4] || 2000), A = 21;
let st = 12345;
const rnd = () => { st ^= st << 13; st >>>= 0; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; };
const seqs = new Int32Array(N * L);
for (let k = 0; k < seqs.length; k++) seqs[k] = (rnd() * A) | 0;
const g = new Gremlin({ L, A, N, seqs, uniformWeights: true, backend: wasm,
                        cfg: { batch: B, lr: Gremlin.suggestLr(L, B) } });

const now = () => performance.now();
const t = { step: 0, data: 0, adam: 0, cmap: 0 };
for (let s = 0; s < 3; s++) g.step();              // warm up
const S = 20;
for (let s = 0; s < S; s++) {
  let t0 = now(); g.step(); t.step += now() - t0;
}
// the same step again, piecewise
for (let s = 0; s < S; s++) {
  const Bn = g._buildBatch();
  let t0 = now();
  g.pll = g.wasm.x.data_pass(g.pW, g.pG, g.pb, g.pGb, g.pLi, g.pSeqs, g.pBatch, g.pCoef, L, A, Bn);
  let t1 = now(); t.data += t1 - t0;
  g._adam(true);                                  // fused symmetrize + Adam
  let t3 = now(); t.adam += t3 - t1;
}
for (let s = 0; s < 5; s++) { let t0 = now(); g.contactMap(); t.cmap += now() - t0; }
const f = (x, n) => (x / n).toFixed(1).padStart(7) + ' ms';
console.log(`L=${L} B=${B} A=${A}  P=${(g.P / 1e6).toFixed(1)}M`);
console.log('step      ' + f(t.step, S) + '   (' + (1000 * S / t.step).toFixed(2) + ' steps/s)');
console.log('  data    ' + f(t.data, S));
console.log('  sym+adam' + f(t.adam, S));
console.log('contact   ' + f(t.cmap, 5));
console.log('loss ' + g.loss.toFixed(6));
