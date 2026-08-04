// Is the fast path memory-bound? Compare sequence-major (random access into W)
// against position-major loop interchange (W swept linearly, small hot buffer).
const A = 21;

function seqMajor(W, bias, x, L, N, G, Gb, lg) {
  G.fill(0); Gb.fill(0); const invN = 1 / N; let loss = 0;
  for (let n = 0; n < N; n++) {
    const s = n * L;
    for (let i = 0; i < L; i++) {
      const bi = i * A, rowI = i * L;
      for (let a = 0; a < A; a++) lg[bi + a] = bias[bi + a];
      for (let j = 0; j < L; j++) { if (j === i) continue;
        const o = ((rowI + j) * A + x[s + j]) * A;
        for (let a = 0; a < A; a++) lg[bi + a] += W[o + a]; }
      let mx = -Infinity; for (let a = 0; a < A; a++) if (lg[bi + a] > mx) mx = lg[bi + a];
      let sum = 0; for (let a = 0; a < A; a++) { const e = Math.exp(lg[bi + a] - mx); lg[bi + a] = e; sum += e; }
      const iv = 1 / sum; for (let a = 0; a < A; a++) lg[bi + a] *= iv;
      loss -= Math.log(lg[bi + x[s + i]]) * invN;
      const xi = x[s + i];
      for (let a = 0; a < A; a++) { const d = (lg[bi + a] - (a === xi ? 1 : 0)) * invN; lg[bi + a] = d; Gb[bi + a] += d; }
      for (let j = 0; j < L; j++) { if (j === i) continue;
        const o = ((rowI + j) * A + x[s + j]) * A;
        for (let a = 0; a < A; a++) G[o + a] += lg[bi + a]; }
    }
  }
  return loss;
}

function posMajor(W, bias, x, L, N, G, Gb, Li) {
  G.fill(0); Gb.fill(0); const invN = 1 / N; let loss = 0;
  for (let i = 0; i < L; i++) {
    const bi = i * A, rowI = i * L;
    for (let n = 0; n < N; n++) { const r = n * A; for (let a = 0; a < A; a++) Li[r + a] = bias[bi + a]; }
    for (let j = 0; j < L; j++) { if (j === i) continue;
      const blk = (rowI + j) * A * A;
      for (let n = 0; n < N; n++) { const o = blk + x[n * L + j] * A, r = n * A;
        for (let a = 0; a < A; a++) Li[r + a] += W[o + a]; } }
    for (let n = 0; n < N; n++) {
      const r = n * A, xi = x[n * L + i];
      let mx = -Infinity; for (let a = 0; a < A; a++) if (Li[r + a] > mx) mx = Li[r + a];
      let sum = 0; for (let a = 0; a < A; a++) { const e = Math.exp(Li[r + a] - mx); Li[r + a] = e; sum += e; }
      const iv = 1 / sum; for (let a = 0; a < A; a++) Li[r + a] *= iv;
      loss -= Math.log(Li[r + xi]) * invN;
      for (let a = 0; a < A; a++) { const d = (Li[r + a] - (a === xi ? 1 : 0)) * invN; Li[r + a] = d; Gb[bi + a] += d; }
    }
    for (let j = 0; j < L; j++) { if (j === i) continue;
      const blk = (rowI + j) * A * A;
      for (let n = 0; n < N; n++) { const o = blk + x[n * L + j] * A, r = n * A;
        for (let a = 0; a < A; a++) G[o + a] += Li[r + a]; } }
  }
  return loss;
}

console.log("    L     N | seq-major ms | pos-major ms | speedup | loss match");
for (const [L, N] of [[32, 500], [64, 1000], [128, 1000], [128, 4000]]) {
  const P = L * L * A * A;
  let st = 5; const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const x = new Int32Array(L * N); for (let k = 0; k < L * N; k++) x[k] = (rnd() * A) | 0;
  const W = new Float32Array(P); for (let k = 0; k < P; k++) W[k] = (rnd() - 0.5) * 0.2;
  const bias = new Float32Array(L * A);
  const G1 = new Float32Array(P), G2 = new Float32Array(P);
  const Gb = new Float32Array(L * A), lg = new Float64Array(L * A), Li = new Float64Array(N * A);
  const t = (f) => { const t0 = process.hrtime.bigint(); const r = f(); return [Number(process.hrtime.bigint() - t0) / 1e6, r]; };
  const [t1, l1] = t(() => seqMajor(W, bias, x, L, N, G1, Gb, lg));
  const [t2, l2] = t(() => posMajor(W, bias, x, L, N, G2, Gb, Li));
  let mg = 0; for (let k = 0; k < P; k++) mg = Math.max(mg, Math.abs(G1[k] - G2[k]));
  console.log("%s %s | %s | %s | %sx | dloss %s  dG %s",
    String(L).padStart(5), String(N).padStart(5), t1.toFixed(0).padStart(12), t2.toFixed(0).padStart(12),
    (t1 / t2).toFixed(1).padStart(6), Math.abs(l1 - l2).toExponential(1), mg.toExponential(1));
}
