/*
 * gremlin.c -- WASM SIMD backend for the hot kernels.
 *
 * Build (see build-wasm.sh; gremlin.wasm is committed so there is no build step
 * for users):
 *   clang --target=wasm32 -O3 -msimd128 -ffast-math -nostdlib \
 *         -Wl,--no-entry -Wl,--export-dynamic -Wl,--import-memory
 *
 * These are transcriptions of the JS in gremlin-core.js, in the same operation
 * order, so the two agree to f32 rounding. They are NOT bit-identical: exp and
 * log are approximated here (see fast_expf / fast_logf) because calling back
 * into JS Math would mean ~417k boundary crossings per step at L=155.
 * Agreement is asserted in test/wasm.test.mjs and enforced again at runtime by
 * the backend self-test in gremlin-core.js.
 *
 * Everything operates on pointers into the single imported linear memory, which
 * JS owns and grows. W, G, m and v live there permanently, and JS holds
 * Float32Array views over the same bytes, so contactMap(), topCouplings(),
 * forward() and the snapshot path keep working unchanged on the same data.
 *
 * No libc: -nostdlib means no malloc, no math.h. sqrt comes from the wasm
 * f32.sqrt instruction via __builtin_sqrtf.
 */

#include <wasm_simd128.h>

#define EXPORT(name) __attribute__((export_name(name)))

typedef unsigned int u32;

/*
 * -nostdlib means no libc, but clang still lowers bulk assignments to memset /
 * memcpy calls. Provide them. no_builtin stops the compiler from recognising
 * these definitions as memset/memcpy and rewriting them into calls to
 * themselves.
 */
__attribute__((no_builtin("memset")))
void *memset(void *dst, int c, unsigned long n) {
  unsigned char *p = (unsigned char *)dst;
  for (unsigned long i = 0; i < n; i++) p[i] = (unsigned char)c;
  return dst;
}

__attribute__((no_builtin("memcpy")))
void *memcpy(void *dst, const void *src, unsigned long n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  for (unsigned long i = 0; i < n; i++) d[i] = s[i];
  return dst;
}

/* ------------------------------------------------------------------ */
/* exp / log                                                          */
/* ------------------------------------------------------------------ */

/*
 * exp for the softmax. Called only on (logit - rowmax), so the argument is
 * always <= 0 and anything below about -88 underflows to zero.
 * exp(x) = 2^k * exp(r) with x = k*ln2 + r, |r| <= ln2/2, then a degree-6
 * Taylor polynomial on r.
 *
 * Measured relative error is ~2e-6, about 17x f32 epsilon. That is deliberate:
 * it is approximate at f32 and that is fine here. These values feed a gradient,
 * not a reported number, and the end-to-end deviation from the JS reference is
 * 9.5e-5 against a 2e-3 acceptance tolerance (see selfTest). Chasing the last
 * few ulps costs f64 ops in the hottest loop and buys nothing measurable.
 */
static inline float fast_expf(float x) {
  if (x < -87.3f) return 0.0f;
  float kf = x * 1.44269504088896341f;              /* x / ln2 */
  int k = (int)(kf - 0.5f);                         /* x <= 0, so kf <= 0 */
  if (kf - (float)k >= 1.0f) k += 1;
  float r = x - (float)k * 0.693147180559945286f;
  /* |r| <= ln2/2 = 0.347. Degree 6: truncation is ~r^7/5040 ~ 1.2e-7, i.e. at
     f32 epsilon. Degree 5 left ~6e-6, which is 50x worse than the format. */
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666666667f
            + r * (0.041666666667f + r * (0.0083333333f + r * 0.0013888889f)))));
  union { float f; u32 i; } u;
  int e = k + 127;
  if (e <= 0) return 0.0f;
  u.i = (u32)e << 23;
  return p * u.f;
}

/*
 * log for the loss term. Only ever called on the softmax denominator, which
 * lies in [1, A] because the largest exponent contributes exactly 1 -- so a
 * narrow-range approximation is plenty. atanh form: log(m) = 2*atanh((m-1)/(m+1)).
 */
static inline float fast_logf(float x) {
  union { float f; u32 i; } u;
  u.f = x;
  int e = (int)((u.i >> 23) & 0xFF) - 127;
  u.i = (u.i & 0x807FFFFFu) | (127u << 23);          /* mantissa into [1,2) */
  float m = u.f;
  float t = (m - 1.0f) / (m + 1.0f);
  float t2 = t * t;
  float lm = 2.0f * t * (1.0f + t2 * (0.33333333f + t2 * (0.2f
             + t2 * (0.14285714f + t2 * 0.11111111f))));
  return lm + (float)e * 0.693147180559945286f;
}

EXPORT("probe_exp") float probe_exp(float x) { return fast_expf(x); }
EXPORT("probe_log") float probe_log(float x) { return fast_logf(x); }

/* ------------------------------------------------------------------ */
/* one optimizer step: forward, softmax, backward                     */
/* ------------------------------------------------------------------ */

/*
 * Mirrors Gremlin.prototype.step's data pass. Position-major: i outer, so the
 * row slab of W (and of G) for position i -- L blocks of A x A, 273KB at L=155
 * -- is what the inner loops touch, and it stays cache-resident.
 *
 * Inside a position the loops run sequence-outer, j-inner. That keeps one
 * sequence's logits in NV = ceil(A/4) vector registers for the whole sweep over
 * j instead of loading and storing Li for every (j, n) -- the j-outer order did
 * twice the memory operations per add. Summation order per element is
 * unchanged (b, then j ascending; backward accumulates n ascending), so the
 * results are bit-identical to the JS reference's ordering.
 *
 * The vector loads are NV*4 wide, i.e. up to 3 floats past the A-row of a
 * block. Reads past the row land in the next row or block of the same array
 * (the last block ever read is (L-1, L-2), never the final one) and only feed
 * lanes that are discarded. Writes in the backward pass add the zero padding
 * lanes of d into those neighbours, which leaves them unchanged.
 *
 * Li has row stride LS = NV*4 and is scratch: logits, then probabilities, then
 * the local gradient d = coef * (p - onehot) with the padding lanes zeroed.
 *
 * Returns the weighted -PLL as a double.
 */
static inline __attribute__((always_inline))
double data_pass_nv(float *W, float *G, float *b, float *Gb, float *Li,
                    const int *seqs, const int *batch, const float *coef,
                    int L, int A, int B, const int NV) {
  const int AA = A * A, LS = NV * 4;
  double loss = 0.0;

  for (int k = 0; k < L * A; k++) Gb[k] = 0.0f;

  for (int i = 0; i < L; i++) {
    const int bi = i * A;
    const float *Wi = W + (unsigned long)i * L * AA;
    float *Gi = G + (unsigned long)i * L * AA;

    /* forward: bias, then one A-vector per j, held in registers */
    for (int n = 0; n < B; n++) {
      const int *sn = seqs + batch[n] * L;
      v128_t acc[6];
      for (int q = 0; q < NV; q++) acc[q] = wasm_v128_load(b + bi + 4 * q);
      for (int j = 0; j < L; j++) {
        if (j == i) continue;
        const float *w = Wi + j * AA + sn[j] * A;
        for (int q = 0; q < NV; q++)
          acc[q] = wasm_f32x4_add(acc[q], wasm_v128_load(w + 4 * q));
      }
      float *dst = Li + n * LS;
      for (int q = 0; q < NV; q++) wasm_v128_store(dst + 4 * q, acc[q]);
    }

    /* stable softmax, loss, then overwrite Li with the local gradient */
    for (int n = 0; n < B; n++) {
      float *row = Li + n * LS;
      const int xi = seqs[batch[n] * L + i];
      const float c = coef[n];

      float mx = row[0];
      for (int a = 1; a < A; a++) if (row[a] > mx) mx = row[a];
      float sum = 0.0f;
      for (int a = 0; a < A; a++) { float e = fast_expf(row[a] - mx); row[a] = e; sum += e; }
      float iv = 1.0f / sum;
      for (int a = 0; a < A; a++) row[a] *= iv;

      float p = row[xi];
      loss -= (double)c * (double)fast_logf(p > 1e-30f ? p : 1e-30f);

      for (int a = 0; a < A; a++) {
        float d = c * (row[a] - (a == xi ? 1.0f : 0.0f));
        row[a] = d;
        Gb[bi + a] += d;
      }
      for (int a = A; a < LS; a++) row[a] = 0.0f;   /* padding adds nothing */
    }

    /* backward: zero this slab of G now, while it is about to be hot, rather
       than in a separate 4*P-byte pass up front */
    for (int k = 0; k < L * AA; k++) Gi[k] = 0.0f;
    for (int n = 0; n < B; n++) {
      const int *sn = seqs + batch[n] * L;
      const float *row = Li + n * LS;
      v128_t d[6];
      for (int q = 0; q < NV; q++) d[q] = wasm_v128_load(row + 4 * q);
      for (int j = 0; j < L; j++) {
        if (j == i) continue;
        float *g = Gi + j * AA + sn[j] * A;
        for (int q = 0; q < NV; q++)
          wasm_v128_store(g + 4 * q, wasm_f32x4_add(wasm_v128_load(g + 4 * q), d[q]));
      }
    }
  }
  return loss;
}

/* One instantiation per register count, so `acc` and `d` are real registers. */
EXPORT("data_pass")
double data_pass(float *W, float *G, float *b, float *Gb, float *Li,
                 const int *seqs, const int *batch, const float *coef,
                 int L, int A, int B) {
  switch ((A + 3) >> 2) {
    case 1: return data_pass_nv(W, G, b, Gb, Li, seqs, batch, coef, L, A, B, 1);
    case 2: return data_pass_nv(W, G, b, Gb, Li, seqs, batch, coef, L, A, B, 2);
    case 3: return data_pass_nv(W, G, b, Gb, Li, seqs, batch, coef, L, A, B, 3);
    case 4: return data_pass_nv(W, G, b, Gb, Li, seqs, batch, coef, L, A, B, 4);
    case 5: return data_pass_nv(W, G, b, Gb, Li, seqs, batch, coef, L, A, B, 5);
    case 6: return data_pass_nv(W, G, b, Gb, Li, seqs, batch, coef, L, A, B, 6);
    default: return -1.0;   /* A > 24: JS refuses this backend (see Gremlin) */
  }
}

/* ------------------------------------------------------------------ */
/* symmetrize                                                         */
/* ------------------------------------------------------------------ */

/*
 * total_ij(a,b) = G_ij(a,b) + G_ji(b,a), written into both halves. Writing the
 * identical value into both is what keeps W exactly symmetric forever.
 */
EXPORT("symmetrize")
void symmetrize(float *G, int L, int A) {
  const int AA = A * A;
  for (int i = 0; i < L; i++) {
    for (int j = i + 1; j < L; j++) {
      float *pij = G + (i * L + j) * AA;
      float *pji = G + (j * L + i) * AA;
      for (int bq = 0; bq < A; bq++) {
        for (int a = 0; a < A; a++) {
          float s = pij[bq * A + a] + pji[a * A + bq];
          pij[bq * A + a] = s;
          pji[a * A + bq] = s;
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* fused Adam + L2                                                    */
/* ------------------------------------------------------------------ */

/*
 * One pass, no allocation, and the sum of squares comes out for free.
 * This loop is O(L^2 A^2) and independent of batch size -- it is the ~100ms
 * per-step floor at L=155 that makes throughput saturate, so it is the single
 * most valuable thing to vectorize.
 */
EXPORT("adam")
double adam(float *W, const float *G, float *M, float *V, int P,
            float lr, float b1, float b2, float eps,
            float ibc1, float ibc2, float gW) {
  const float om1 = 1.0f - b1, om2 = 1.0f - b2;
  double sumsq = 0.0;
  for (int k = 0; k < P; k++) {
    float w = W[k];
    sumsq += (double)w * (double)w;
    float g = G[k] + gW * w;
    float m = b1 * M[k] + om1 * g; M[k] = m;
    float v = b2 * V[k] + om2 * g * g; V[k] = v;
    W[k] = w - lr * (m * ibc1) / (__builtin_sqrtf(v * ibc2) + eps);
  }
  return sumsq;
}

/*
 * symmetrize + Adam + L2 in one pass over the i<j blocks only.
 *
 * W, G, m and v are all exactly symmetric (W_ji = W_ij^T), so updating both
 * halves elementwise did every Adam computation twice. Here each pair of blocks
 * is symmetrized while it is hot, Adam runs on the i<j block -- contiguous, so
 * it vectorizes -- and the result is transposed into the j<i block. m and v are
 * only kept for i<j; their lower halves are never read. Diagonal blocks are
 * zero and stay zero (zero gradient, zero moments), so they are skipped.
 *
 * Every element goes through the same f32 operations in the same order as the
 * scalar `adam` above -- f32x4 sqrt and div are correctly rounded, and wasm
 * never fuses a multiply-add -- so W is bit-identical to symmetrize + adam.
 * Only the reported sum of squares is summed in a different order.
 */
EXPORT("sym_adam")
double sym_adam(float *W, float *G, float *M, float *V, int L, int A,
                float lr, float b1, float b2, float eps,
                float ibc1, float ibc2, float gW) {
  const int AA = A * A, AA4 = AA & ~3;
  const float om1 = 1.0f - b1, om2 = 1.0f - b2;
  const v128_t vb1 = wasm_f32x4_splat(b1), vb2 = wasm_f32x4_splat(b2);
  const v128_t vom1 = wasm_f32x4_splat(om1), vom2 = wasm_f32x4_splat(om2);
  const v128_t vlr = wasm_f32x4_splat(lr), veps = wasm_f32x4_splat(eps);
  const v128_t vi1 = wasm_f32x4_splat(ibc1), vi2 = wasm_f32x4_splat(ibc2);
  const v128_t vgW = wasm_f32x4_splat(gW);
  v128_t s2 = wasm_f64x2_splat(0.0);
  double tail = 0.0;

  for (int i = 0; i < L; i++) {
    for (int j = i + 1; j < L; j++) {
      const unsigned long oIJ = ((unsigned long)i * L + j) * AA;
      const unsigned long oJI = ((unsigned long)j * L + i) * AA;
      float *gij = G + oIJ, *gji = G + oJI;
      float *wij = W + oIJ, *wji = W + oJI;
      float *mij = M + oIJ, *vij = V + oIJ;

      for (int bq = 0; bq < A; bq++) {
        for (int a = 0; a < A; a++) {
          float s = gij[bq * A + a] + gji[a * A + bq];
          gij[bq * A + a] = s;
          gji[a * A + bq] = s;
        }
      }

      int k = 0;
      for (; k < AA4; k += 4) {
        v128_t w = wasm_v128_load(wij + k);
        s2 = wasm_f64x2_add(s2, wasm_f64x2_mul(wasm_f64x2_promote_low_f32x4(w),
                                               wasm_f64x2_promote_low_f32x4(w)));
        v128_t wh = wasm_i32x4_shuffle(w, w, 2, 3, 0, 1);
        s2 = wasm_f64x2_add(s2, wasm_f64x2_mul(wasm_f64x2_promote_low_f32x4(wh),
                                               wasm_f64x2_promote_low_f32x4(wh)));
        v128_t g = wasm_f32x4_add(wasm_v128_load(gij + k), wasm_f32x4_mul(vgW, w));
        v128_t m = wasm_f32x4_add(wasm_f32x4_mul(vb1, wasm_v128_load(mij + k)),
                                  wasm_f32x4_mul(vom1, g));
        wasm_v128_store(mij + k, m);
        v128_t v = wasm_f32x4_add(wasm_f32x4_mul(vb2, wasm_v128_load(vij + k)),
                                  wasm_f32x4_mul(wasm_f32x4_mul(vom2, g), g));
        wasm_v128_store(vij + k, v);
        v128_t up = wasm_f32x4_div(wasm_f32x4_mul(vlr, wasm_f32x4_mul(m, vi1)),
                                   wasm_f32x4_add(wasm_f32x4_sqrt(wasm_f32x4_mul(v, vi2)), veps));
        wasm_v128_store(wij + k, wasm_f32x4_sub(w, up));
      }
      for (; k < AA; k++) {
        float w = wij[k];
        tail += (double)w * (double)w;
        float g = gij[k] + gW * w;
        float m = b1 * mij[k] + om1 * g; mij[k] = m;
        float v = b2 * vij[k] + om2 * g * g; vij[k] = v;
        wij[k] = w - lr * (m * ibc1) / (__builtin_sqrtf(v * ibc2) + eps);
      }

      for (int bq = 0; bq < A; bq++)
        for (int a = 0; a < A; a++)
          wji[a * A + bq] = wij[bq * A + a];
    }
  }
  return 2.0 * (wasm_f64x2_extract_lane(s2, 0) + wasm_f64x2_extract_lane(s2, 1) + tail);
}

/* ------------------------------------------------------------------ */
/* GREMLIN_TF's modified Adam                                         */
/* ------------------------------------------------------------------ */

/*
 * GREMLIN_TF v2.1 replaces Adam's per-element second moment with a single
 * scalar per tensor -- the running mean of the squared gradient *norm* -- and
 * disables bias correction:
 *
 *   vt <- b2*vt + (1-b2)*sum(g*g)        (scalar)
 *   x  <- x - lr/(sqrt(vt)+eps) * mt
 *
 * Every element shares one normalizer, so the update keeps the direction of the
 * gradient and only its overall scale is standardized. Per-element Adam instead
 * divides each component by its own |g|, which amplifies components whose
 * gradient is near zero all the way to +-lr.
 *
 * Two passes are needed because the norm must be known before any element is
 * updated: grad_norm2 reduces, adam_scaled applies.
 */
EXPORT("grad_norm2")
double grad_norm2(const float *G, const float *W, int P, float gW) {
  double s = 0.0;
  for (int k = 0; k < P; k++) {
    float g = G[k] + gW * W[k];
    s += (double)g * (double)g;
  }
  return s;
}

EXPORT("adam_scaled")
double adam_scaled(float *W, const float *G, float *M, int P,
                   float lrEff, float b1, float gW) {
  const float om1 = 1.0f - b1;
  double sumsq = 0.0;
  for (int k = 0; k < P; k++) {
    float w = W[k];
    sumsq += (double)w * (double)w;
    float g = G[k] + gW * w;
    float m = b1 * M[k] + om1 * g;
    M[k] = m;
    W[k] = w - lrEff * m;
  }
  return sumsq;
}

/* ------------------------------------------------------------------ */
/* Meff neighbour counts                                              */
/* ------------------------------------------------------------------ */

/*
 * The exact definition: cnt[n] = |{m : identity(n,m) >= threshold}|, from
 * which w_n = 1/cnt[n] and Meff = sum w_n. It is O(N^2 L) and it is the whole
 * startup cost on a real alignment -- 28s in JS on a 15688 x 155 MSA, which is
 * why the JS path approximates above maxRefs.
 *
 * There is nothing to approximate if the exact count is cheap enough, and it
 * can be: the comparison is a byte equality test, so i8x16_eq plus bitmask
 * plus popcount does sixteen positions in three instructions. Rows are stored
 * 16-byte aligned (stride = (L + 15) & ~15) with the padding zeroed, so the
 * loads never straddle a row and the padding always matches -- the tail is
 * corrected once, outside the loop, rather than branched around inside it.
 *
 * The early exit survives: once the mismatches make `need` unreachable the
 * pair cannot count, so bail. It is tested per 16-byte block instead of per
 * 4 bytes, which is a coarser bail but a far cheaper loop.
 *
 * Runs over rows [nFrom, nTo) so JS can slice the pass and report progress;
 * every slice accumulates into the same cnt array, and because the inner loop
 * starts at n+1 and bumps both counters, slicing by rows still visits each
 * unordered pair exactly once.
 */
EXPORT("meff_counts")
void meff_counts(const unsigned char *seq, float *cnt, int N, int stride,
                 int L, int need, int nFrom, int nTo) {
  const int blocks = (L + 15) >> 4;          // 16-byte blocks covering L

  for (int n = nFrom; n < nTo; n++) {
    const unsigned char *a = seq + (unsigned long)n * stride;
    for (int m = n + 1; m < N; m++) {
      const unsigned char *b = seq + (unsigned long)m * stride;
      int raw = 0, k = 0;
      for (int q = 0; q < blocks; q++, k += 16) {
        v128_t e = wasm_i8x16_eq(wasm_v128_load(a + k), wasm_v128_load(b + k));
        raw += __builtin_popcount((unsigned)wasm_i8x16_bitmask(e));
        /*
         * `raw` counts the zero padding too, and padding always matches, so the
         * finished real count is raw - (blocks*16 - L). The best still
         * reachable after this block is that plus every remaining byte:
         * raw + (blocks*16 - k - 16) - (blocks*16 - L) = raw + L - k - 16.
         * On the last block the term is exactly the finished count, so this one
         * test serves as both the early exit and the acceptance test -- falling
         * out of the loop already means the pair qualifies.
         */
        if (raw + (L - k - 16) < need) goto next;
      }
      cnt[n] += 1.0f; cnt[m] += 1.0f;
      next:;
    }
  }
}

/* ------------------------------------------------------------------ */
/* contact map                                                        */
/* ------------------------------------------------------------------ */

/*
 * Zero-sum gauge per A x A block, Frobenius norm, then APC. Writes an L x L
 * matrix. Runs once per snapshot (~10 Hz) and is O(L^2 A^2): ~49ms at L=155
 * in JS, ~13ms as scalar WASM, a few ms vectorized.
 * `scratch` must hold at least 2*A floats.
 */
EXPORT("contact_map")
void contact_map(const float *W, float *out, float *scratch, int L, int A, int nA) {
  const int AA = A * A;
  float *rowM = scratch;
  float *colM = scratch + A;
  /* nA excludes the gap state: the reference takes the norm over the 20x20
     amino-acid block only. Blocks are still strided by the full A. */
  const float invA = 1.0f / (float)nA;
  const float invAA = 1.0f / (float)(nA * nA);
  const int nV = nA >> 2 > 8 ? 8 : nA >> 2, nV4 = nV * 4;   /* whole vectors in a row */

  for (int i = 0; i < L; i++) out[i * L + i] = 0.0f;

  for (int i = 0; i < L; i++) {
    for (int j = i + 1; j < L; j++) {
      const float *blk = W + (i * L + j) * AA;
      /* row sums vectorize over a (nV full vectors, scalar tail); column sums
         are horizontal sums of each row. Same quantities as the JS reference,
         summed in a different order, so agreement is to f32 rounding. */
      v128_t rv[8];
      for (int q = 0; q < nV; q++) rv[q] = wasm_f32x4_splat(0.0f);
      for (int a = nV4; a < nA; a++) rowM[a] = 0.0f;
      float all = 0.0f;
      for (int bq = 0; bq < nA; bq++) {
        const float *r = blk + bq * A;
        v128_t cs = wasm_f32x4_splat(0.0f);
        for (int q = 0; q < nV; q++) {
          v128_t w = wasm_v128_load(r + 4 * q);
          rv[q] = wasm_f32x4_add(rv[q], w);
          cs = wasm_f32x4_add(cs, w);
        }
        float c = wasm_f32x4_extract_lane(cs, 0) + wasm_f32x4_extract_lane(cs, 1)
                + wasm_f32x4_extract_lane(cs, 2) + wasm_f32x4_extract_lane(cs, 3);
        for (int a = nV4; a < nA; a++) { rowM[a] += r[a]; c += r[a]; }
        colM[bq] = c * invA;
        all += c;
      }
      for (int q = 0; q < nV; q++) wasm_v128_store(rowM + 4 * q, wasm_f32x4_mul(rv[q], wasm_f32x4_splat(invA)));
      for (int a = nV4; a < nA; a++) rowM[a] *= invA;
      all *= invAA;
      v128_t sv = wasm_f32x4_splat(0.0f);
      float s = 0.0f;
      for (int bq = 0; bq < nA; bq++) {
        const float *r = blk + bq * A;
        const float off = all - colM[bq];
        const v128_t voff = wasm_f32x4_splat(off);
        for (int q = 0; q < nV; q++) {
          v128_t w = wasm_f32x4_add(wasm_f32x4_sub(wasm_v128_load(r + 4 * q),
                                                   wasm_v128_load(rowM + 4 * q)), voff);
          sv = wasm_f32x4_add(sv, wasm_f32x4_mul(w, w));
        }
        for (int a = nV4; a < nA; a++) { float w = r[a] - rowM[a] + off; s += w * w; }
      }
      s += wasm_f32x4_extract_lane(sv, 0) + wasm_f32x4_extract_lane(sv, 1)
         + wasm_f32x4_extract_lane(sv, 2) + wasm_f32x4_extract_lane(sv, 3);
      float f = __builtin_sqrtf(s);
      out[i * L + j] = f;
      out[j * L + i] = f;
    }
  }

  /* average product correction (Dunn): S_ij - S_i. S_.j / S_.. */
  float *rs = scratch;                      /* reuse: needs L floats now */
  double tot = 0.0;
  for (int i = 0; i < L; i++) {
    float r = 0.0f;
    for (int j = 0; j < L; j++) r += out[i * L + j];
    rs[i] = r;
    tot += (double)r;
  }
  const float invTot = (float)(1.0 / (tot + 1e-8));
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < L; j++) {
      out[i * L + j] = (i == j) ? 0.0f : out[i * L + j] - rs[i] * rs[j] * invTot;
    }
  }
}

/* ------------------------------------------------------------------ */
/* forward pass for one sequence (network panel / snapshots)           */
/* ------------------------------------------------------------------ */

EXPORT("forward_one")
void forward_one(const float *W, const float *b, const int *seqs,
                 float *logits, float *probs, int L, int A, int sel) {
  const int AA = A * A;
  const int s = sel * L;
  for (int i = 0; i < L; i++) {
    const int bi = i * A, rowI = i * L;
    for (int a = 0; a < A; a++) logits[bi + a] = b[bi + a];
    for (int j = 0; j < L; j++) {
      if (j == i) continue;
      const float *w = W + (rowI + j) * AA + seqs[s + j] * A;
      for (int a = 0; a < A; a++) logits[bi + a] += w[a];
    }
    float mx = logits[bi];
    for (int a = 1; a < A; a++) if (logits[bi + a] > mx) mx = logits[bi + a];
    float sum = 0.0f;
    for (int a = 0; a < A; a++) { float e = fast_expf(logits[bi + a] - mx); probs[bi + a] = e; sum += e; }
    float iv = 1.0f / sum;
    for (int a = 0; a < A; a++) probs[bi + a] *= iv;
  }
}

/* ------------------------------------------------------------------ */
/* memory bookkeeping                                                 */
/* ------------------------------------------------------------------ */

extern unsigned char __heap_base;

/** First byte JS may use for its own arrays. */
EXPORT("heap_base") int heap_base(void) { return (int)&__heap_base; }

EXPORT("zero") void zero(float *p, int n) { for (int k = 0; k < n; k++) p[k] = 0.0f; }
