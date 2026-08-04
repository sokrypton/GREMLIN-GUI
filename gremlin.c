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
 * Mirrors Gremlin.prototype.step's data pass. Position-major: i outer,
 * sequences inner, so W streams linearly and Li (B x A) stays hot. The inner
 * loops over `a` are contiguous and vectorize to f32x4.
 *
 * Li is reused as scratch: it holds the bias, then the logits, then the
 * probabilities, then the local gradient d = coef * (p - onehot).
 *
 * Returns the weighted -PLL as a double.
 */
EXPORT("data_pass")
double data_pass(float *W, float *G, float *b, float *Gb, float *Li,
                 const int *seqs, const int *batch, const float *coef,
                 int L, int A, int B) {
  const int AA = A * A;
  double loss = 0.0;

  for (int k = 0; k < L * L * AA; k++) G[k] = 0.0f;
  for (int k = 0; k < L * A; k++) Gb[k] = 0.0f;

  for (int i = 0; i < L; i++) {
    const int bi = i * A, rowI = i * L;

    /* forward: seed with the bias, then add one contiguous A-vector per j */
    for (int n = 0; n < B; n++) {
      float *dst = Li + n * A;
      const float *src = b + bi;
      for (int a = 0; a < A; a++) dst[a] = src[a];
    }
    for (int j = 0; j < L; j++) {
      if (j == i) continue;
      const int blk = (rowI + j) * AA;
      for (int n = 0; n < B; n++) {
        const float *w = W + blk + seqs[batch[n] * L + j] * A;
        float *dst = Li + n * A;
        for (int a = 0; a < A; a++) dst[a] += w[a];
      }
    }

    /* stable softmax, loss, then overwrite Li with the local gradient */
    for (int n = 0; n < B; n++) {
      float *row = Li + n * A;
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
    }

    /* backward: same access pattern as forward, so W and G both stream */
    for (int j = 0; j < L; j++) {
      if (j == i) continue;
      const int blk = (rowI + j) * AA;
      for (int n = 0; n < B; n++) {
        float *g = G + blk + seqs[batch[n] * L + j] * A;
        const float *row = Li + n * A;
        for (int a = 0; a < A; a++) g[a] += row[a];
      }
    }
  }
  return loss;
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

/* ------------------------------------------------------------------ */
/* contact map                                                        */
/* ------------------------------------------------------------------ */

/*
 * Zero-sum gauge per A x A block, Frobenius norm, then APC. Writes an L x L
 * matrix. Runs once per snapshot and is O(L^2 A^2), ~49ms at L=155 in JS.
 * `scratch` must hold at least 2*A floats.
 */
EXPORT("contact_map")
void contact_map(const float *W, float *out, float *scratch, int L, int A) {
  const int AA = A * A;
  float *rowM = scratch;
  float *colM = scratch + A;
  const float invA = 1.0f / (float)A;
  const float invAA = 1.0f / (float)AA;

  for (int i = 0; i < L; i++) out[i * L + i] = 0.0f;

  for (int i = 0; i < L; i++) {
    for (int j = i + 1; j < L; j++) {
      const float *blk = W + (i * L + j) * AA;
      for (int a = 0; a < A; a++) { rowM[a] = 0.0f; colM[a] = 0.0f; }
      float all = 0.0f;
      for (int bq = 0; bq < A; bq++) {
        for (int a = 0; a < A; a++) {
          float w = blk[bq * A + a];
          rowM[a] += w;                     /* sum over b, for state a at i */
          colM[bq] += w;                    /* sum over a, for state b at j */
          all += w;
        }
      }
      for (int a = 0; a < A; a++) { rowM[a] *= invA; colM[a] *= invA; }
      all *= invAA;
      float s = 0.0f;
      for (int bq = 0; bq < A; bq++) {
        for (int a = 0; a < A; a++) {
          float w = blk[bq * A + a] - rowM[a] - colM[bq] + all;
          s += w * w;
        }
      }
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
