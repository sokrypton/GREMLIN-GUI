// Benchmark: the original index.html hot loop vs. the one-hot formulation.
// Verifies the fast path reproduces the naive gradient/loss exactly, then times both.

//////////////////// NAIVE (verbatim port of index.html) ////////////////////

const getSymmetricWeight = (weights, i, j, L, A) => {
  const fromL = Math.floor(i / A), toL = Math.floor(j / A);
  return fromL === toL ? 0 : weights[fromL < toL ? i * L * A + j : j * L * A + i];
};

const softmaxNaive = (arr) => {
  const expValues = arr.map(Math.exp);
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  return expValues.map((v) => v / sumExp);
};

function calculateModelOutputs(weights, bias, sample, L, A) {
  const x = sample.flatMap((s) => Array(A).fill(0).map((_, i) => (i === s ? 1 : 0)));
  const logits = Array(L * A).fill(0).map((_, i) => {
    const weightedSum = x.reduce(
      (sum, val, j) =>
        sum + (Math.floor(i / A) !== Math.floor(j / A) ? val * getSymmetricWeight(weights, i, j, L, A) : 0),
      0
    );
    return weightedSum + bias[i];
  });
  const x_ = Array(L).fill(null).flatMap((_, i) => softmaxNaive(logits.slice(i * A, (i + 1) * A)));
  return { oneHot: x, preSoftmax: logits, postSoftmax: x_ };
}

// The gradient accumulation loop from optimizeStep, verbatim in structure.
function naiveGrads(weights, bias, seqs, L, A) {
  const LA = L * A;
  let weightGrads = Array(LA * LA).fill(0);
  let biasGrads = Array(LA).fill(0);
  let totalLoss = 0;
  const numSamples = seqs.length;

  seqs.forEach((sample) => {
    const { oneHot, postSoftmax } = calculateModelOutputs(weights, bias, sample, L, A);
    for (let i = 0; i < LA; i++) {
      for (let j = i + 1; j < LA; j++) {
        if (Math.floor(i / A) !== Math.floor(j / A)) {
          const gradient = (postSoftmax[i] - oneHot[i]) * oneHot[j] + (postSoftmax[j] - oneHot[j]) * oneHot[i];
          weightGrads[i * LA + j] += gradient / numSamples;
          weightGrads[j * LA + i] += gradient / numSamples;
        }
      }
      biasGrads[i] += (postSoftmax[i] - oneHot[i]) / numSamples;
    }
    const sampleLoss = -oneHot.reduce((sum, val, i) => sum + (val ? Math.log(postSoftmax[i]) : 0), 0);
    totalLoss += sampleLoss / numSamples;
  });
  return { weightGrads, biasGrads, totalLoss };
}

//////////////////// FAST (blocked typed arrays, one-hot sparsity) ////////////////////
// Layout: W[((i*L + j)*A + b)*A + a]  = coupling contributing to logit_i(a) when x_j == b.
// Symmetry invariant: W[(i,j,b,a)] === W[(j,i,a,b)].
// Inner loop over `a` is contiguous => vectorizable, and we never touch the
// (LA)^2 dense product because x is one-hot (only b = x[j] is live).

function fastStep(W, bias, seqsFlat, L, A, N, G, Gb, logits) {
  G.fill(0); Gb.fill(0);
  const invN = 1 / N;
  let totalLoss = 0;

  for (let n = 0; n < N; n++) {
    const s = n * L;
    // ---- forward ----
    for (let i = 0; i < L; i++) {
      const bi = i * A;
      for (let a = 0; a < A; a++) logits[bi + a] = bias[bi + a];
      const rowI = i * L;
      for (let j = 0; j < L; j++) {
        if (j === i) continue;
        const off = ((rowI + j) * A + seqsFlat[s + j]) * A;
        for (let a = 0; a < A; a++) logits[bi + a] += W[off + a];
      }
      // ---- stable softmax in place ----
      let mx = -Infinity;
      for (let a = 0; a < A; a++) if (logits[bi + a] > mx) mx = logits[bi + a];
      let sum = 0;
      for (let a = 0; a < A; a++) { const e = Math.exp(logits[bi + a] - mx); logits[bi + a] = e; sum += e; }
      const inv = 1 / sum;
      for (let a = 0; a < A; a++) logits[bi + a] *= inv;
      totalLoss -= Math.log(logits[bi + seqsFlat[s + i]]) * invN;
    }
    // ---- gradient ----
    for (let i = 0; i < L; i++) {
      const bi = i * A, xi = seqsFlat[s + i], rowI = i * L;
      for (let a = 0; a < A; a++) {
        const d = (logits[bi + a] - (a === xi ? 1 : 0)) * invN;
        logits[bi + a] = d;
        Gb[bi + a] += d;
      }
      for (let j = 0; j < L; j++) {
        if (j === i) continue;
        const off = ((rowI + j) * A + seqsFlat[s + j]) * A;
        for (let a = 0; a < A; a++) G[off + a] += logits[bi + a];
      }
    }
  }
  return totalLoss;
}

// Enforce the symmetry of the coupling gradient: total_ij(a,b) = G_ij(a,b) + G_ji(b,a)
function symmetrize(G, L, A) {
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      const oIJ = (i * L + j) * A * A, oJI = (j * L + i) * A * A;
      for (let b = 0; b < A; b++) {
        for (let a = 0; a < A; a++) {
          const s = G[oIJ + b * A + a] + G[oJI + a * A + b];
          G[oIJ + b * A + a] = s; G[oJI + a * A + b] = s;
        }
      }
    }
  }
}

//////////////////// setup helpers ////////////////////

function makeProblem(L, A, N, seed) {
  let st = seed >>> 0;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const seqs = [], seqsFlat = new Int32Array(L * N);
  for (let n = 0; n < N; n++) {
    const row = [];
    for (let i = 0; i < L; i++) { const v = (rnd() * A) | 0; row.push(v); seqsFlat[n * L + i] = v; }
    seqs.push(row);
  }
  const LA = L * A;
  // A random *symmetric* coupling tensor, written into both representations.
  const Wnaive = Array(LA * LA).fill(0);
  const Wfast = new Float64Array(L * L * A * A);
  const bias = Array(LA).fill(0).map(() => (rnd() - 0.5) * 0.4);
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++) {
      const c = (rnd() - 0.5) * 0.6;
      Wnaive[(i * A + a) * LA + (j * A + b)] = c;   // canonical half
      Wnaive[(j * A + b) * LA + (i * A + a)] = c;   // mirror, as optimizeStep maintains
      Wfast[((i * L + j) * A + b) * A + a] = c;
      Wfast[((j * L + i) * A + a) * A + b] = c;
    }
  return { seqs, seqsFlat, Wnaive, Wfast, bias, LA };
}

//////////////////// 1. correctness ////////////////////

{
  const L = 6, A = 5, N = 7;
  const p = makeProblem(L, A, N, 42);
  const nv = naiveGrads(p.Wnaive, p.bias, p.seqs, L, A);
  const G = new Float64Array(L * L * A * A), Gb = new Float64Array(L * A), lg = new Float64Array(L * A);
  const fastLoss = fastStep(p.Wfast, p.bias, p.seqsFlat, L, A, N, G, Gb, lg);
  symmetrize(G, L, A);

  let maxG = 0, maxB = 0;
  for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++)
    for (let a = 0; a < A; a++) for (let b = 0; b < A; b++)
      maxG = Math.max(maxG, Math.abs(
        nv.weightGrads[(i * A + a) * p.LA + (j * A + b)] - G[((i * L + j) * A + b) * A + a]));
  for (let k = 0; k < L * A; k++) maxB = Math.max(maxB, Math.abs(nv.biasGrads[k] - Gb[k]));

  console.log("=== correctness (L=6 A=5 N=7) ===");
  console.log("  loss  naive %s  fast %s   |diff| %s",
    nv.totalLoss.toFixed(12), fastLoss.toFixed(12), Math.abs(nv.totalLoss - fastLoss).toExponential(2));
  console.log("  max |dW diff| %s      max |db diff| %s", maxG.toExponential(2), maxB.toExponential(2));
}

//////////////////// 2. naive vs fast, same size ////////////////////

function time(fn, reps = 1) {
  const t0 = process.hrtime.bigint();
  for (let r = 0; r < reps; r++) fn();
  return Number(process.hrtime.bigint() - t0) / 1e6 / reps;
}

console.log("\n=== naive vs fast, one full-batch step (ms) ===");
console.log("    L    A     N |     naive |      fast | speedup");
for (const [L, A, N] of [[3, 3, 4], [8, 21, 32], [16, 21, 64], [24, 21, 128], [32, 21, 256]]) {
  const p = makeProblem(L, A, N, 7);
  const G = new Float64Array(L * L * A * A), Gb = new Float64Array(L * A), lg = new Float64Array(L * A);
  const reps = L <= 8 ? 5 : 1;
  const tN = time(() => naiveGrads(p.Wnaive, p.bias, p.seqs, L, A), reps);
  const Wf = new Float32Array(p.Wfast); const Gf = new Float32Array(G.length);
  const tF = time(() => { fastStep(Wf, p.bias, p.seqsFlat, L, A, N, Gf, Gb, lg); symmetrize(Gf, L, A); }, 5);
  console.log("%s %s %s | %s | %s | %sx",
    String(L).padStart(5), String(A).padStart(4), String(N).padStart(5),
    tN.toFixed(1).padStart(9), tF.toFixed(1).padStart(9), (tN / tF).toFixed(0).padStart(6));
}

//////////////////// 3. fast path at realistic scale ////////////////////

console.log("\n=== fast path only, realistic sizes (A=21, f32) ===");
console.log("    L      N | step ms | steps/s |  params | W+G+m+v MB");
for (const [L, N] of [[64, 1000], [128, 1000], [128, 5000], [256, 1000], [256, 5000], [384, 2000]]) {
  const A = 21, P = L * L * A * A;
  const mb = (P * 4 * 4) / 1048576;
  if (mb > 3000) { console.log("  skip L=%d (%d MB)", L, mb | 0); continue; }
  const seqsFlat = new Int32Array(L * N);
  let st = 3; const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let k = 0; k < L * N; k++) seqsFlat[k] = (rnd() * A) | 0;
  const W = new Float32Array(P), G = new Float32Array(P);
  for (let k = 0; k < P; k++) W[k] = (rnd() - 0.5) * 0.2;
  const bias = new Float32Array(L * A), Gb = new Float32Array(L * A), lg = new Float64Array(L * A);
  const t = time(() => { fastStep(W, bias, seqsFlat, L, A, N, G, Gb, lg); symmetrize(G, L, A); }, 2);
  console.log("%s %s | %s | %s | %s | %s",
    String(L).padStart(5), String(N).padStart(6), t.toFixed(0).padStart(7),
    (1000 / t).toFixed(2).padStart(7), (P / 1e6).toFixed(1).padStart(6) + "M",
    mb.toFixed(0).padStart(10));
}

//////////////////// 4. render-side element counts ////////////////////

console.log("\n=== SVG elements the current render emits ===");
console.log("    L    A |  network <line> |  weight <rect> | contact <rect>");
for (const [L, A] of [[3, 3], [10, 10], [64, 21], [128, 21], [256, 21]]) {
  const LA = L * A;
  console.log("%s %s | %s | %s | %s",
    String(L).padStart(5), String(A).padStart(4),
    (L * (L - 1) * A * A).toLocaleString().padStart(15),
    (LA * LA).toLocaleString().padStart(14), (L * L).toLocaleString().padStart(14));
}
