/*
 * The original index.html implementation, extracted verbatim (modulo being
 * parameterized on L/A instead of closing over React state) so the rewritten
 * core can be checked against it rather than against my own re-derivation.
 *
 * Cost is O(L^2 A^2) per sequence in the forward pass and another O(L^2 A^2)
 * in the gradient loop; both collapse to O(L^2 A) once one-hot sparsity is
 * used. Kept only as a test oracle.
 */

export const getSymmetricWeight = (weights, i, j, L, A) => {
  const fromL = Math.floor(i / A), toL = Math.floor(j / A);
  return fromL === toL ? 0 : weights[fromL < toL ? i * L * A + j : j * L * A + i];
};

const softmax = (arr) => {
  const expValues = arr.map(Math.exp);
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  return expValues.map((v) => v / sumExp);
};

export function calculateModelOutputs(weights, bias, sample, L, A) {
  const x = sample.flatMap((s) => Array(A).fill(0).map((_, i) => (i === s ? 1 : 0)));
  const logits = Array(L * A).fill(0).map((_, i) => {
    const weightedSum = x.reduce(
      (sum, val, j) =>
        sum + (Math.floor(i / A) !== Math.floor(j / A)
          ? val * getSymmetricWeight(weights, i, j, L, A) : 0),
      0
    );
    return weightedSum + bias[i];
  });
  const x_ = Array(L).fill(null).flatMap((_, i) => softmax(logits.slice(i * A, (i + 1) * A)));
  return { oneHot: x, preSoftmax: logits, postSoftmax: x_ };
}

/** Unregularized data gradient and mean PLL, exactly as optimizeStep computed them. */
export function naiveGrads(weights, bias, seqs, L, A) {
  const LA = L * A;
  const weightGrads = Array(LA * LA).fill(0);
  const biasGrads = Array(LA).fill(0);
  let totalLoss = 0;
  const numSamples = seqs.length;

  seqs.forEach((sample) => {
    const { oneHot, postSoftmax } = calculateModelOutputs(weights, bias, sample, L, A);
    for (let i = 0; i < LA; i++) {
      for (let j = i + 1; j < LA; j++) {
        if (Math.floor(i / A) !== Math.floor(j / A)) {
          const gradient = (postSoftmax[i] - oneHot[i]) * oneHot[j]
                         + (postSoftmax[j] - oneHot[j]) * oneHot[i];
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
