/*
 * gremlin-core.js -- numeric core for GREMLIN (pseudo-likelihood Potts / plmDCA).
 *
 * Loads two ways:
 *   - as a Web Worker script  (new Worker('gremlin-core.js'))  -> message plumbing at the bottom
 *   - as a plain script / node import                          -> globalThis.GremlinCore
 * No DOM, no dependencies, so the numerics are testable outside the browser.
 *
 * ---------------------------------------------------------------------------
 * Storage layout
 * ---------------------------------------------------------------------------
 * W[((i*L + j)*A + b)*A + a]  = coupling added to logit_i(a) when position j
 *                               holds state b.
 *
 * Two properties make this layout worth the index arithmetic:
 *   1. The innermost loop runs over `a` contiguously, so it vectorizes and the
 *      whole A-vector for a given (i,j,b) is one cache line burst.
 *   2. Because the input is one-hot, only b = x_j is ever live. The forward
 *      pass is therefore O(L^2 A), not O(L^2 A^2) -- an A-fold (21x) saving
 *      over materializing the dense one-hot vector and reducing over it.
 *
 * Symmetry invariant:  W[(i,j,b,a)] === W[(j,i,a,b)].
 * The diagonal blocks (i === j) are allocated but permanently zero; they cost
 * L*A^2 of L^2*A^2 (0.8% at L=128) and buying them back would complicate every
 * index expression. Packing the i<j half would halve memory -- see the note on
 * `bytes()` below.
 *
 * ---------------------------------------------------------------------------
 * Loop order
 * ---------------------------------------------------------------------------
 * Steps iterate position-major (i outer, sequences inner) rather than
 * sequence-major. Sequence-major touches L*(L-1) scattered A x A blocks per
 * sequence, so it streams the whole of W once per sequence and is bound by
 * cache misses, not arithmetic. Position-major reads W linearly and keeps a
 * B x A scratch buffer hot instead; measured 2.3x on top of the sparsity win,
 * with bit-identical gradients.
 *
 * ---------------------------------------------------------------------------
 * Objective
 * ---------------------------------------------------------------------------
 *   J/Meff = (1/Meff) sum_n w_n sum_i -log p_i(x_ni)
 *            + (lam_w/Meff) sum W^2  +  (lam_b/Meff) sum b^2
 *
 *   lam_w = alpha * (L-1) * (A-1)      alpha default 0.01
 *   lam_b = beta                       beta  default 0.01
 *
 * The L-scaling on lam_w is what makes a single regularization knob behave the
 * same at L=8 and L=256; a bare constant (as in the original) is only ever
 * tuned for one size. `sum W^2` runs over the full symmetric tensor, so each
 * unique coupling is counted twice -- this matches GREMLIN's convention and is
 * consistent with the gradient (2*lam_w*W applied to each of the two copies).
 *
 * Minibatching decouples cost from N: sequences are sampled with probability
 * w_n/Meff and each carries coefficient 1/B, which is an unbiased estimator of
 * the Meff-normalized mean above. N=100k costs the same per step as N=256.
 */
(function (root) {
  'use strict';

  var IS_WORKER = typeof self !== 'undefined' && typeof self.importScripts === 'function';

  /* ------------------------------------------------------------------ */
  /* helpers                                                            */
  /* ------------------------------------------------------------------ */

  // Numerically stable softmax of src[off .. off+A) written into dst[off .. off+A).
  // The original omitted the max subtraction, which turns into Infinity/NaN once
  // logits grow -- reachable on real data with small alpha.
  function softmaxRange(src, dst, off, A) {
    var mx = -Infinity, a, e, s = 0;
    for (a = 0; a < A; a++) if (src[off + a] > mx) mx = src[off + a];
    for (a = 0; a < A; a++) { e = Math.exp(src[off + a] - mx); dst[off + a] = e; s += e; }
    s = 1 / s;
    for (a = 0; a < A; a++) dst[off + a] *= s;
  }

  var LOG_FLOOR = 1e-30;

  /* ------------------------------------------------------------------ */
  /* WASM backend                                                       */
  /* ------------------------------------------------------------------ */

  /*
   * gremlin.wasm holds the same kernels as the JS below, compiled with
   * -msimd128. The parameters live in WASM linear memory and JS keeps
   * Float32Array views over the same bytes, so everything that only *reads* the
   * parameters -- topCouplings, couplingMatrix, block, the snapshot path --
   * keeps working unchanged. Only the hot loops are replaced.
   *
   * Growing WASM memory detaches existing views, so the whole layout is sized
   * and reserved up front and never grown again.
   */
  function WasmBackend(instance, memory) {
    this.name = 'wasm';
    this.x = instance.exports;
    this.memory = memory;
    this.ptr = this.x.heap_base();
  }

  WasmBackend.prototype.reserve = function (bytes) {
    var need = this.ptr + bytes + 65536;
    var have = this.memory.buffer.byteLength;
    if (need > have) {
      var pages = Math.ceil((need - have) / 65536);
      this.memory.grow(pages);              // invalidates every existing view
    }
    this.base = this.memory.buffer;
  };

  /* Bump-allocate a view. Called only between reserve() and first use. */
  WasmBackend.prototype.f32 = function (n) {
    var p = this.ptr;
    this.ptr += n * 4;
    return { p: p, v: new Float32Array(this.base, p, n) };
  };
  WasmBackend.prototype.i32 = function (n) {
    var p = this.ptr;
    this.ptr += n * 4;
    return { p: p, v: new Int32Array(this.base, p, n) };
  };

  /*
   * Fetch and instantiate gremlin.wasm. Memory is imported rather than exported
   * so JS controls growth. Returns null on any failure, which is a normal
   * outcome -- the caller falls back to JS.
   */
  function initWasm(url) {
    if (typeof WebAssembly === 'undefined') return Promise.resolve(null);
    var memory;
    try {
      memory = new WebAssembly.Memory({ initial: 16 });
    } catch (e) { return Promise.resolve(null); }

    // Browser/worker: fetch. node (no `self`): read from disk, for the tests.
    var inBrowser = typeof self !== 'undefined' && typeof fetch === 'function';
    var get;
    if (inBrowser) {
      get = fetch(url).then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' fetching ' + url);
        return r.arrayBuffer();
      });
    } else if (typeof require === 'function') {
      try {
        get = Promise.resolve(require('fs').readFileSync(url));
      } catch (e) { get = Promise.reject(e); }
    } else {
      API.lastWasmError = 'no way to load ' + url;
      return Promise.resolve(null);
    }

    return get.then(function (buf) {
      return WebAssembly.instantiate(buf, { env: { memory: memory } });
    }).then(function (res) {
      return new WasmBackend(res.instance, memory);
    }).catch(function (e) {
      // A failure here is normal (no WASM, blocked fetch); the caller falls back
      // to JS. Record why, so it can be reported rather than guessed at.
      API.lastWasmError = String(e && e.message || e);
      return null;
    });
  }

  /* ------------------------------------------------------------------ */
  /* Gremlin                                                            */
  /* ------------------------------------------------------------------ */

  /**
   * opts = {
   *   L, A, N,
   *   seqs      : Int32Array(N*L), state index per (sequence, position)
   *   sw        : Float32Array(N) sequence weights (optional; computed if absent)
   *   identity  : identity threshold for reweighting (default 0.8)
   *   maxRefs   : above this N, reweighting is approximated (default 3000)
   *   cfg       : { alpha, beta, lr, batch }
   *   seed      : PRNG seed
   * }
   */
  function Gremlin(opts) {
    this.L = opts.L | 0;
    this.A = opts.A | 0;
    this.N = opts.N | 0;
    this.AA = this.A * this.A;
    this.P = this.L * this.L * this.AA;
    this.seqs = opts.seqs;

    /*
     * With a WASM backend the parameters must live in its linear memory. Size
     * the whole layout first, reserve once, then bump-allocate -- growing later
     * would detach every view.
     */
    var LA = this.L * this.A;
    var bk = opts.backend || null;
    var be = bk && bk.name === 'wasm' ? bk : null;
    this.wasm = be;
    this.gpu = bk && bk.name === 'webgpu' ? bk : null;

    /*
     * Batch buffers are allocated once at the largest batch the UI can request,
     * so changing the batch size later never reallocates -- which matters
     * because bump-allocating into WASM memory a second time would leak, and
     * growing that memory would detach every view.
     */
    var Bmax = Math.min(Math.max(4096, (opts.cfg && opts.cfg.batch) | 0), this.N);
    if (Bmax < 1) Bmax = 1;
    this.Bmax = Bmax;

    if (be) {
      var scratchN = Math.max(2 * this.A, this.L);
      var floats = 4 * this.P + 4 * LA + Bmax * this.A + scratchN
                 + this.L * this.L + 2 * LA + Bmax;
      var ints = this.N * this.L + Bmax;
      be.reserve(floats * 4 + ints * 4);

      var w = be.f32(this.P); this.W = w.v; this.pW = w.p;
      var g = be.f32(this.P); this.G = g.v; this.pG = g.p;
      var m = be.f32(this.P); this.M = m.v; this.pM = m.p;
      var v = be.f32(this.P); this.V = v.v; this.pV = v.p;
      var bb = be.f32(LA); this.b = bb.v; this.pb = bb.p;
      var gb = be.f32(LA); this.Gb = gb.v; this.pGb = gb.p;
      var mb = be.f32(LA); this.Mb = mb.v; this.pMb = mb.p;
      var vb = be.f32(LA); this.Vb = vb.v; this.pVb = vb.p;
      var sc = be.f32(scratchN); this.scratch = sc.v; this.pScratch = sc.p;
      var co = be.f32(this.L * this.L); this.cmOut = co.v; this.pCmOut = co.p;
      var lo = be.f32(LA); this.loBuf = lo.v; this.pLo = lo.p;
      var pr = be.f32(LA); this.prBuf = pr.v; this.pPr = pr.p;
      var sq = be.i32(this.N * this.L); this.seqs = sq.v; this.pSeqs = sq.p;
      this.seqs.set(opts.seqs);
      var li = be.f32(Bmax * this.A); this.Li = li.v; this.pLi = li.p;
      var ba = be.i32(Bmax); this.batch = ba.v; this.pBatch = ba.p;
      var cf = be.f32(Bmax); this.coef = cf.v; this.pCoef = cf.p;
    } else {
      this.W = new Float32Array(this.P);
      this.G = new Float32Array(this.P);
      this.M = new Float32Array(this.P);
      this.V = new Float32Array(this.P);
      this.b = new Float32Array(LA);
      this.Gb = new Float32Array(LA);
      this.Mb = new Float32Array(LA);
      this.Vb = new Float32Array(LA);
      this.Li = new Float32Array(Bmax * this.A);
      this.batch = new Int32Array(Bmax);
      this.coef = new Float32Array(Bmax);
    }

    /*
     * regMode picks how alpha/beta become penalties:
     *   'gremlin' (default) lam_w = alpha*(L-1)*(A-1)/Meff, lam_b = beta/Meff
     *              -- L-scaled and Meff-normalized, so one setting transfers
     *                 across alignment sizes. Use this for real data.
     *   'raw'      lam_w = alpha/2, lam_b = beta/2, no L or Meff scaling, which
     *              makes the gradient exactly alpha*W + data -- the original
     *              index.html convention. The educational page uses this so its
     *              slider numbers still mean what they used to.
     */
    this.cfg = {
      alpha: 0.01, beta: 0.01, lr: 0.05, batch: 256,
      b1: 0.9, b2: 0.999, eps: 1e-8, regMode: 'gremlin', optMode: 'adam'
    };
    if (opts.cfg) for (var k in opts.cfg) if (opts.cfg[k] !== undefined) this.cfg[k] = opts.cfg[k];

    this.rng = (opts.seed | 0) || 1234567;
    this.t = 0;
    this.steps = 0;
    this.vt = 0;
    this.vtb = 0;
    this.pll = 0;
    this.regW = 0;
    this.regB = 0;
    this.loss = 0;
    this.rms = 0;
    this.sel = 0;
    this.approxWeights = false;

    if (opts.uniformWeights) {
      // No reweighting: every sequence counts once, Meff = N. The original had
      // no notion of sequence weights, so the educational page opts out to stay
      // numerically comparable.
      this.sw = new Float32Array(this.N);
      this.sw.fill(1);
      this.Meff = this.N;
    } else if (opts.sw) {
      this.sw = opts.sw;
      var s = 0;
      for (var n = 0; n < this.N; n++) s += this.sw[n];
      this.Meff = s;
    } else {
      this.computeWeights(opts.identity === undefined ? 0.8 : opts.identity,
                          opts.maxRefs || 3000, opts.onProgress);
    }
    /*
     * Which states the contact norm covers. The reference takes the Frobenius
     * norm over the 20x20 amino-acid block only -- "note: we ignore gaps" --
     * because gap couplings carry alignment and phylogeny signal rather than
     * structural contact. Our alphabet puts the gap last, so this is a prefix.
     */
    this.gap = opts.gap === undefined ? (this.A === 21 ? 20 : -1) : opts.gap;
    this.normA = (this.gap === this.A - 1) ? this.A - 1 : this.A;

    this.biasInit = opts.biasInit === undefined ? 'freq' : opts.biasInit;

    this._allocBatch();
    this.resetHistory();
    if (this.biasInit === 'freq') this.initBias();

    /*
     * With WebGPU the parameters live on the device. The CPU arrays above become
     * a shadow that syncW() refreshes, and only for models small enough that the
     * readback is trivial -- above that the shadow is dropped entirely, because
     * every consumer of it (the node diagram, the coupling matrix, the top-K
     * scan) is illegible at that size and gated off anyway.
     */
    if (this.gpu) {
      this.shadow = this.P <= this.gpu.SYNC_MAX;
      if (!this.shadow) {
        this.W = new Float32Array(0);
        this.G = new Float32Array(0);
        this.M = new Float32Array(0);
        this.V = new Float32Array(0);
      }
      this.gpu.setup(this);
    }
  }

  /*
   * Start the bias at the single-site log frequencies rather than at zero, as
   * the reference does:
   *     b_ini = log(sum(X, 0) + 0.01*log(N));  b = b_ini - mean(b_ini, -1)
   * The conditionals then begin already explaining the per-column composition,
   * so the couplings do not have to spend early steps absorbing single-site
   * signal. Counts are unweighted, matching the reference.
   */
  Gremlin.prototype.initBias = function () {
    var L = this.L, A = this.A, N = this.N, seqs = this.seqs, b = this.b;
    var pseudo = 0.01 * Math.log(Math.max(N, 2));
    var counts = new Float64Array(L * A);
    var n, i, a, off, mean;
    for (n = 0; n < N; n++) {
      off = n * L;
      for (i = 0; i < L; i++) counts[i * A + seqs[off + i]] += 1;
    }
    for (i = 0; i < L; i++) {
      mean = 0;
      for (a = 0; a < A; a++) {
        b[i * A + a] = Math.log(counts[i * A + a] + pseudo);
        mean += b[i * A + a];
      }
      mean /= A;
      for (a = 0; a < A; a++) b[i * A + a] -= mean;
    }
  };

  /*
   * The reference's learning-rate heuristic: 0.1 * log(batch) / L. It has to
   * shrink with L because the pseudo-likelihood sums L conditionals per
   * sequence. A fixed rate that works at L=48 overshoots badly at L=155.
   */
  Gremlin.suggestLr = function (L, B) {
    return 0.1 * Math.log(Math.max(B, 2)) / Math.max(L, 1);
  };

  // Bytes held by the parameter set: W + G + m + v. This, not FLOPs, is the
  // ceiling on L in a browser tab -- full-rank Potts is inherently O(L^2 A^2).
  // Packing the i<j half would halve it (L=384: 992MB -> 520MB).
  Gremlin.bytes = function (L, A) { return 4 * 4 * L * L * A * A; };

  Gremlin.prototype.bytes = function () { return Gremlin.bytes(this.L, this.A); };

  Gremlin.prototype._rand = function () {
    var x = this.rng;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    this.rng = x;
    return x / 4294967296;
  };

  /* ---------------- sequence reweighting (Meff) --------------------- */

  /*
   * w_n = 1 / |{m : identity(n,m) >= threshold}|,  Meff = sum_n w_n.
   *
   * This is O(N^2 L) and is the step people forget when they wonder why a fast
   * solver still takes a minute to start. Two mitigations here: an early exit
   * once the remaining positions cannot reach the threshold, and, above
   * maxRefs, counting neighbours against a random reference subset and scaling
   * (approximate, flagged via this.approxWeights).
   */
  Gremlin.prototype.computeWeights = function (identity, maxRefs, onProgress) {
    var N = this.N, L = this.L;
    var sw = new Float32Array(N), n;
    this.identity = identity;

    if (N <= 1) {
      if (N === 1) sw[0] = 1;
      this.sw = sw; this.Meff = N; return;
    }

    var need = Math.ceil(identity * L);
    var cnt = new Float32Array(N);

    /*
     * The comparison runs over a byte-packed copy rather than the Int32 states.
     * This loop is memory-bound over pairs, so 1 byte per residue instead of 4
     * quarters the traffic, and XOR-ing 32 bits at a time compares four
     * positions per operation. Measured on a 15688 x 155 alignment: 17.9s down
     * to a couple of seconds, which is the difference between a usable first
     * load and one that looks hung.
     */
    var words = (L / 4) | 0;              // whole 4-residue words
    var tail = L - words * 4;             // leftover residues, compared bytewise
    var stride = words * 4 + (tail ? 4 : 0);
    var bytes = new Uint8Array(N * stride);
    var seqs = this.seqs, k;
    for (n = 0; n < N; n++) {
      for (k = 0; k < L; k++) bytes[n * stride + k] = seqs[n * L + k];
    }
    var u32 = new Uint32Array(bytes.buffer);
    var wStride = stride >> 2;

    // identity(n, m) >= need ?
    function near(an, am) {
      var id = 0, w, x, remaining;
      for (w = 0; w < words; w++) {
        x = u32[an + w] ^ u32[am + w];
        if (x === 0) { id += 4; continue; }
        if ((x & 0xff) === 0) id++;
        if ((x & 0xff00) === 0) id++;
        if ((x & 0xff0000) === 0) id++;
        if ((x & 0xff000000) === 0) id++;
        remaining = L - (w + 1) * 4;
        if (id + remaining < need) return false;          // unreachable, bail
      }
      if (tail) {
        x = u32[an + words] ^ u32[am + words];
        if ((tail > 0) && (x & 0xff) === 0) id++;
        if ((tail > 1) && (x & 0xff00) === 0) id++;
        if ((tail > 2) && (x & 0xff0000) === 0) id++;
      }
      return id >= need;
    }

    if (N <= maxRefs) {
      for (n = 0; n < N; n++) cnt[n] = 1;                 // each sequence counts itself
      for (n = 0; n < N; n++) {
        var an = n * wStride;
        for (var m = n + 1; m < N; m++) {
          if (near(an, m * wStride)) { cnt[n]++; cnt[m]++; }
        }
        if (onProgress && (n & 255) === 0) onProgress(n / N);
      }
      this.approxWeights = false;
    } else {
      /*
       * Above maxRefs, count neighbours against a random reference subset and
       * scale by N/R. Note this is biased: 1/cnt is convex, so a noisier count
       * inflates Meff (measured 8099 at R=3000 versus 11406 at R=500 on the
       * same alignment). Larger R is both slower and more accurate.
       */
      var R = maxRefs, refs = new Int32Array(R), r;
      for (r = 0; r < R; r++) refs[r] = (this._rand() * N) | 0;
      var scale = N / R;
      for (n = 0; n < N; n++) {
        var an2 = n * wStride, hits = 0;
        for (r = 0; r < R; r++) {
          if (refs[r] === n) continue;
          if (near(an2, refs[r] * wStride)) hits++;
        }
        cnt[n] = 1 + hits * scale;
        if (onProgress && (n & 255) === 0) onProgress(n / N);
      }
      this.approxWeights = true;
    }

    var tot = 0;
    for (n = 0; n < N; n++) { sw[n] = 1 / cnt[n]; tot += sw[n]; }
    this.sw = sw;
    this.Meff = tot;
  };

  /* ---------------- batching ---------------------------------------- */

  /* Only picks the working batch size and rebuilds the sampling table; the
     buffers themselves were sized once, in the constructor. */
  Gremlin.prototype._allocBatch = function () {
    var B = Math.min(this.cfg.batch | 0 || 1, this.N, this.Bmax);
    if (B < 1) B = 1;
    this.B = B;
    var cum = new Float32Array(this.N), acc = 0;
    for (var n = 0; n < this.N; n++) { acc += this.sw[n]; cum[n] = acc; }
    this.cum = cum;
    this.cumTot = acc;
  };

  Gremlin.prototype.setConfig = function (cfg) {
    var reBatch = cfg.batch !== undefined && (cfg.batch | 0) !== this.cfg.batch;
    for (var k in cfg) if (cfg[k] !== undefined) this.cfg[k] = cfg[k];
    if (reBatch) this._allocBatch();
  };

  Gremlin.prototype._buildBatch = function () {
    var N = this.N, B = this.B, batch = this.batch, coef = this.coef, n;
    if (B >= N) {                                  // full batch
      var inv = 1 / this.Meff;
      for (n = 0; n < N; n++) { batch[n] = n; coef[n] = this.sw[n] * inv; }
      return N;
    }
    var cum = this.cum, c = 1 / B, lo, hi, mid, r;
    for (n = 0; n < B; n++) {                      // sample proportional to w_n
      r = this._rand() * this.cumTot;
      lo = 0; hi = N - 1;
      while (lo < hi) { mid = (lo + hi) >> 1; if (cum[mid] < r) lo = mid + 1; else hi = mid; }
      batch[n] = lo; coef[n] = c;
    }
    return B;
  };

  /* ---------------- one optimizer step ------------------------------ */

  Gremlin.prototype.step = function () {
    var L = this.L, A = this.A, AA = this.AA;
    var W = this.W, G = this.G, b = this.b, Gb = this.Gb;
    var Li = this.Li, batch = this.batch, coef = this.coef, seqs = this.seqs;
    var B = this._buildBatch();
    var i, j, n, a, k, r, o, bi, rowI, blk, xi, c, mx, s, iv, e, d;

    if (this.wasm) {
      this.pll = this.wasm.x.data_pass(
        this.pW, this.pG, this.pb, this.pGb, this.pLi,
        this.pSeqs, this.pBatch, this.pCoef, L, A, B);
      this._symmetrize();
      this._adam();
      this.steps++;
      return this.loss;
    }

    G.fill(0); Gb.fill(0);
    var loss = 0;

    for (i = 0; i < L; i++) {
      bi = i * A; rowI = i * L;

      /* forward: seed with the bias, then accumulate one contiguous A-vector
         per (j, x_j). Only b = x_j is touched -- the one-hot saving. */
      for (n = 0; n < B; n++) { r = n * A; for (a = 0; a < A; a++) Li[r + a] = b[bi + a]; }
      for (j = 0; j < L; j++) {
        if (j === i) continue;
        blk = (rowI + j) * AA;
        for (n = 0; n < B; n++) {
          o = blk + seqs[batch[n] * L + j] * A; r = n * A;
          for (a = 0; a < A; a++) Li[r + a] += W[o + a];
        }
      }

      /* stable softmax, weighted loss, then overwrite Li with the local
         gradient d = coef * (p - onehot) so the backward sweep can reuse it */
      for (n = 0; n < B; n++) {
        r = n * A; xi = seqs[batch[n] * L + i]; c = coef[n];
        mx = -Infinity;
        for (a = 0; a < A; a++) if (Li[r + a] > mx) mx = Li[r + a];
        s = 0;
        for (a = 0; a < A; a++) { e = Math.exp(Li[r + a] - mx); Li[r + a] = e; s += e; }
        iv = 1 / s;
        for (a = 0; a < A; a++) Li[r + a] *= iv;
        e = Li[r + xi];
        loss -= c * Math.log(e > LOG_FLOOR ? e : LOG_FLOOR);
        for (a = 0; a < A; a++) {
          d = c * (Li[r + a] - (a === xi ? 1 : 0));
          Li[r + a] = d;
          Gb[bi + a] += d;
        }
      }

      /* backward: same access pattern as forward, so W and G both stream */
      for (j = 0; j < L; j++) {
        if (j === i) continue;
        blk = (rowI + j) * AA;
        for (n = 0; n < B; n++) {
          o = blk + seqs[batch[n] * L + j] * A; r = n * A;
          for (a = 0; a < A; a++) G[o + a] += Li[r + a];
        }
      }
    }

    this._symmetrize();
    this.pll = loss;
    this._adam();
    this.steps++;
    return this.loss;
  };

  /*
   * The pseudo-likelihood gradient for coupling (i,a)-(j,b) collects a term
   * from position i's conditional and one from position j's, so the two stored
   * copies must both receive the sum. Writing the identical value into both
   * halves is what keeps W exactly symmetric forever: G is symmetric, hence m
   * and v are, hence the elementwise Adam update is -- bit-for-bit, in f32.
   */
  Gremlin.prototype._symmetrize = function () {
    var L = this.L, A = this.A, AA = this.AA, G = this.G;
    var i, j, a, bq, oIJ, oJI, s;
    if (this.wasm) { this.wasm.x.symmetrize(this.pG, L, A); return; }
    for (i = 0; i < L; i++) {
      for (j = i + 1; j < L; j++) {
        oIJ = (i * L + j) * AA; oJI = (j * L + i) * AA;
        for (bq = 0; bq < A; bq++) {
          for (a = 0; a < A; a++) {
            s = G[oIJ + bq * A + a] + G[oJI + a * A + bq];
            G[oIJ + bq * A + a] = s;
            G[oJI + a * A + bq] = s;
          }
        }
      }
    }
  };

  /* Fused in-place Adam + L2. One pass, zero allocation, and the L2 sums come
     out for free -- the original allocated six full-size arrays per step. */
  Gremlin.prototype._adam = function () {
    var cfg = this.cfg, lr = cfg.lr, b1 = cfg.b1, b2 = cfg.b2, eps = cfg.eps;
    var W = this.W, G = this.G, M = this.M, V = this.V, P = this.P;
    var k, w, g, m, v;

    this.t++;
    var ibc1 = 1 / (1 - Math.pow(b1, this.t));
    var ibc2 = 1 / (1 - Math.pow(b2, this.t));
    var om1 = 1 - b1, om2 = 1 - b2;

    /*
     * The 0.5 matches the reference implementation (sokrypton/laxy
     * examples/gremlin_jax.ipynb), which uses
     *     l2 = 0.5*(L-1)*(A-1)*sum(w^2) + sum(b^2),  loss = cce.sum() + lam*l2
     * with lam = 0.01. Its objective is the *sum* over sequences and ours is the
     * Meff-normalized mean, so ours is exactly theirs divided by Meff -- same
     * minimizer, and Adam is invariant to the constant. Without the 0.5 our
     * coupling penalty was twice the reference's at the same alpha.
     */
    var raw = cfg.regMode === 'raw';
    var lamW = raw ? cfg.alpha / 2
                   : 0.5 * cfg.alpha * (this.L - 1) * (this.A - 1) / this.Meff;
    var gW = 2 * lamW, sw2 = 0;
    var lamB = raw ? cfg.beta / 2 : cfg.beta / this.Meff;
    var gB = 2 * lamB, sb2 = 0;
    var b = this.b, Gb = this.Gb, Mb = this.Mb, Vb = this.Vb, nb = b.length;

    /*
     * GREMLIN_TF's optimizer: one scalar second moment per tensor rather than
     * one per element, and no bias correction. See grad_norm2 in gremlin.c.
     */
    if (cfg.optMode === 'gremlin') {
      var n2, n2b;
      if (this.wasm) {
        n2 = this.wasm.x.grad_norm2(this.pG, this.pW, P, gW);
        n2b = this.wasm.x.grad_norm2(this.pGb, this.pb, nb, gB);
      } else {
        n2 = 0;
        for (k = 0; k < P; k++) { g = G[k] + gW * W[k]; n2 += g * g; }
        n2b = 0;
        for (k = 0; k < nb; k++) { g = Gb[k] + gB * b[k]; n2b += g * g; }
      }
      this.vt = b2 * (this.vt || 0) + om2 * n2;
      this.vtb = b2 * (this.vtb || 0) + om2 * n2b;
      var lrW = lr / (Math.sqrt(this.vt) + eps);
      var lrB = lr / (Math.sqrt(this.vtb) + eps);

      if (this.wasm) {
        sw2 = this.wasm.x.adam_scaled(this.pW, this.pG, this.pM, P, lrW, b1, gW);
        sb2 = this.wasm.x.adam_scaled(this.pb, this.pGb, this.pMb, nb, lrB, b1, gB);
      } else {
        for (k = 0; k < P; k++) {
          w = W[k]; sw2 += w * w;
          g = G[k] + gW * w;
          m = b1 * M[k] + om1 * g; M[k] = m;
          W[k] = w - lrW * m;
        }
        for (k = 0; k < nb; k++) {
          w = b[k]; sb2 += w * w;
          g = Gb[k] + gB * w;
          m = b1 * Mb[k] + om1 * g; Mb[k] = m;
          b[k] = w - lrB * m;
        }
      }
      this._finishAdam(lamW, lamB, sw2, sb2);
      return;
    }

    if (this.wasm) {
      sw2 = this.wasm.x.adam(this.pW, this.pG, this.pM, this.pV, P,
                             lr, b1, b2, eps, ibc1, ibc2, gW);
    } else {
      for (k = 0; k < P; k++) {
        w = W[k];
        sw2 += w * w;
        g = G[k] + gW * w;
        m = b1 * M[k] + om1 * g; M[k] = m;
        v = b2 * V[k] + om2 * g * g; V[k] = v;
        W[k] = w - lr * (m * ibc1) / (Math.sqrt(v * ibc2) + eps);
      }
    }

    for (k = 0; k < nb; k++) {
      w = b[k];
      sb2 += w * w;
      g = Gb[k] + gB * w;
      m = b1 * Mb[k] + om1 * g; Mb[k] = m;
      v = b2 * Vb[k] + om2 * g * g; Vb[k] = v;
      b[k] = w - lr * (m * ibc1) / (Math.sqrt(v * ibc2) + eps);
    }

    this._finishAdam(lamW, lamB, sw2, sb2);
  };

  /* Shared tail: report the penalty terms and record the loss point. */
  Gremlin.prototype._finishAdam = function (lamW, lamB, sw2, sb2) {
    this.regW = lamW * sw2;
    this.regB = lamB * sb2;
    // sums are taken pre-update, so they pair with the pll from the same
    // parameters (the original mixed pre-update pll with post-update L2)
    this.loss = this.pll + this.regW + this.regB;
    var live = this.L * (this.L - 1) * this.AA;
    this.rms = live > 0 ? Math.sqrt(sw2 / live) : 0;
    this.recordHistory();
  };

  /* ---------------- async surface (used by the worker) -------------- */

  /*
   * One uniform async API over all three backends, so the worker's loop does not
   * branch on which one is active. JS and WASM resolve immediately; WebGPU
   * encodes and submits without blocking, and the queue is drained by the
   * periodic contactMapAsync() readback.
   */
  Gremlin.prototype.stepAsync = function () {
    if (!this.gpu) return Promise.resolve(this.step());
    var self = this;
    var B = this._buildBatch();
    var cfg = this.cfg;
    this.t++;
    var ibc1 = 1 / (1 - Math.pow(cfg.b1, this.t));
    var ibc2 = 1 / (1 - Math.pow(cfg.b2, this.t));
    var raw = cfg.regMode === 'raw';
    var lamW = raw ? cfg.alpha / 2 : cfg.alpha * (this.L - 1) * (this.A - 1) / this.Meff;
    var lamB = raw ? cfg.beta / 2 : cfg.beta / this.Meff;
    this._lamW = lamW; this._lamB = lamB;
    return this.gpu.step(this, {
      lr: cfg.lr, ibc1: ibc1, ibc2: ibc2, gW: 2 * lamW, gWb: 2 * lamB
    }).then(function () {
      self.steps++;
      return self.loss;
    });
  };

  /* Refresh the reported loss breakdown. Costs three small readbacks, so the
     worker only calls it when it is about to post a snapshot. */
  Gremlin.prototype.syncStats = function () {
    if (!this.gpu) return Promise.resolve();
    var self = this;
    return this.gpu.syncScalars(this).then(function (s) {
      self.pll = s.pll;
      self.regW = self._lamW * s.sw2;
      self.regB = self._lamB * s.sb2;
      self.loss = self.pll + self.regW + self.regB;
      var live = self.L * (self.L - 1) * self.AA;
      self.rms = live > 0 ? Math.sqrt(s.sw2 / live) : 0;
      self.recordHistory();
    });
  };

  Gremlin.prototype.contactMapAsync = function () {
    if (!this.gpu) return Promise.resolve(this.contactMap());
    return this.gpu.contactMap(this);
  };

  Gremlin.prototype.blockAsync = function (i, j) {
    if (!this.gpu) return Promise.resolve(this.block(i, j));
    return this.gpu.blockAt(this, i, j);
  };

  /* Make the CPU-side helpers usable. Resolves false when the model is too big
     to shadow, in which case the caller must skip those panels. */
  Gremlin.prototype.syncParams = function () {
    if (!this.gpu) return Promise.resolve(true);
    return this.gpu.syncW(this);
  };

  Gremlin.prototype.reset = function () {
    this.W.fill(0); this.G.fill(0); this.M.fill(0); this.V.fill(0);
    this.b.fill(0); this.Gb.fill(0); this.Mb.fill(0); this.Vb.fill(0);
    if (this.gpu) this.gpu.reset(this);
    this.t = 0; this.steps = 0; this.vt = 0; this.vtb = 0;
    this.pll = 0; this.regW = 0; this.regB = 0; this.loss = 0; this.rms = 0;
    this.resetHistory();
  };

  /* ---------------- loss history (bounded) -------------------------- */

  /*
   * Fixed-capacity, decimating. The original did setLoss(prev => [...prev, x])
   * every step: O(steps^2) copying and unbounded growth. Here the buffer never
   * exceeds CAP; when it fills, every other point is dropped and the stride
   * doubles, so the curve keeps its full time span at halved resolution.
   */
  var HIST_CAP = 4096;

  Gremlin.prototype.resetHistory = function () {
    this.hist = new Float32Array(HIST_CAP * 2);
    this.histLen = 0;
    this.histStride = 1;
  };

  Gremlin.prototype.recordHistory = function () {
    if (this.steps % this.histStride !== 0) return;
    if (this.histLen >= HIST_CAP) {
      var h = this.hist, w = 0;
      for (var r = 0; r < this.histLen; r += 2) {
        h[w * 2] = h[r * 2]; h[w * 2 + 1] = h[r * 2 + 1]; w++;
      }
      this.histLen = w;
      this.histStride *= 2;
    }
    this.hist[this.histLen * 2] = this.steps;
    this.hist[this.histLen * 2 + 1] = this.loss;
    this.histLen++;
  };

  Gremlin.prototype.history = function () { return this.hist.slice(0, this.histLen * 2); };

  /* ---------------- contact map ------------------------------------- */

  /*
   * Zero-sum (Ising) gauge per A x A block, Frobenius norm, then APC.
   *
   * The gauge fixing is not cosmetic: the coupling parameterization has an
   * A-fold redundancy per block, so the raw Frobenius norm of an unconstrained
   * fit mixes real signal with whatever gauge the optimizer happened to drift
   * into. The original took the norm directly.
   */
  Gremlin.prototype.contactMap = function () {
    var L = this.L, A = this.A, AA = this.AA, W = this.W, nA = this.normA;
    if (this.wasm) {
      this.wasm.x.contact_map(this.pW, this.pCmOut, this.pScratch, L, A, nA);
      return this.cmOut.slice(0);           // detach from WASM memory for transfer
    }
    var F = new Float32Array(L * L);
    var rowM = new Float64Array(A), colM = new Float64Array(A);
    var i, j, a, bq, o, w, all, s, f;
    var nAA = nA * nA;

    for (i = 0; i < L; i++) {
      for (j = i + 1; j < L; j++) {
        o = (i * L + j) * AA;
        rowM.fill(0); colM.fill(0); all = 0;
        // nA excludes the gap state; see this.normA
        for (bq = 0; bq < nA; bq++) {
          for (a = 0; a < nA; a++) {
            w = W[o + bq * A + a];
            rowM[a] += w;                       // sum over b, for state a at i
            colM[bq] += w;                      // sum over a, for state b at j
            all += w;
          }
        }
        for (a = 0; a < nA; a++) { rowM[a] /= nA; colM[a] /= nA; }
        all /= nAA;
        s = 0;
        for (bq = 0; bq < nA; bq++) {
          for (a = 0; a < nA; a++) {
            w = W[o + bq * A + a] - rowM[a] - colM[bq] + all;
            s += w * w;
          }
        }
        f = Math.sqrt(s);
        F[i * L + j] = f; F[j * L + i] = f;
      }
    }

    // average product correction (Dunn): S_ij - S_i. S_.j / S_..
    var rs = new Float64Array(L), tot = 0, rsum;
    for (i = 0; i < L; i++) {
      rsum = 0;
      for (j = 0; j < L; j++) rsum += F[i * L + j];
      rs[i] = rsum; tot += rsum;
    }
    var out = new Float32Array(L * L);
    for (i = 0; i < L; i++) {
      for (j = 0; j < L; j++) {
        out[i * L + j] = i === j ? 0 : F[i * L + j] - rs[i] * rs[j] / (tot + 1e-8);
      }
    }
    return out;
  };

  /** Ranked contacts, {i, j, score} in model column indices, |i-j| >= minSep. */
  Gremlin.prototype.topContacts = function (minSep, limit, cm) {
    var L = this.L;
    cm = cm || this.contactMap();
    var out = [];
    for (var i = 0; i < L; i++) {
      for (var j = i + minSep; j < L; j++) out.push({ i: i, j: j, score: cm[i * L + j] });
    }
    out.sort(function (p, q) { return q.score - p.score; });
    return limit ? out.slice(0, limit) : out;
  };

  /* ---------------- rendering done here, not on the main thread ----- */

  /*
   * W is 110MB at L=128 and 441MB at L=256, so it must never be posted to the
   * main thread. The worker rasterizes it instead and transfers ~1MB of RGBA.
   * Above maxSize the image is max-|.| pooled, which preserves isolated strong
   * couplings that mean-pooling would wash out.
   */
  Gremlin.prototype.couplingImage = function (maxSize, scale) {
    var L = this.L, A = this.A, AA = this.AA, W = this.W;
    var LA = L * A;
    var S = Math.min(LA, (maxSize | 0) || 512);
    var data = new Uint8ClampedArray(S * S * 4);
    var sc = scale || 4 * this.rms;
    var inv = sc > 0 ? 1 / sc : 0;
    var p, q, I, J, I0, I1, J0, J1, i, j, a, bq, rowI, v, av, best, bestAbs, u, o, g;

    for (p = 0; p < S; p++) {
      I0 = Math.floor(p * LA / S); I1 = Math.max(I0 + 1, Math.floor((p + 1) * LA / S));
      for (q = 0; q < S; q++) {
        J0 = Math.floor(q * LA / S); J1 = Math.max(J0 + 1, Math.floor((q + 1) * LA / S));
        best = 0; bestAbs = -1;
        for (I = I0; I < I1; I++) {
          i = (I / A) | 0; a = I - i * A; rowI = i * L;
          for (J = J0; J < J1; J++) {
            j = (J / A) | 0;
            if (j === i) continue;                       // diagonal blocks unused
            bq = J - j * A;
            v = W[(rowI + j) * AA + bq * A + a];
            av = v < 0 ? -v : v;
            if (av > bestAbs) { bestAbs = av; best = v; }
          }
        }
        u = best * inv;
        if (u < -1) u = -1; else if (u > 1) u = 1;
        o = (p * S + q) * 4;
        if (u < 0) { g = Math.round(255 * (1 + u)); data[o] = 255; data[o + 1] = g; data[o + 2] = g; }
        else { g = Math.round(255 * (1 - u)); data[o] = g; data[o + 1] = g; data[o + 2] = 255; }
        data[o + 3] = 255;
      }
    }
    return { data: data, n: S, scale: sc };
  };

  /*
   * Top-K couplings by |w| via a size-K min-heap: O(P) comparisons but only
   * O(K log K) heap work, because almost every candidate fails the single
   * compare against the heap root. The original built an object per coupling
   * and full-sorted all L(L-1)A^2 of them on every render.
   * Returns a flat Float32Array of [i, a, j, b, w] quintuples.
   */
  Gremlin.prototype.topCouplings = function (K) {
    var L = this.L, A = this.A, AA = this.AA, W = this.W;
    var total = (L * (L - 1) / 2) * AA;
    var cap = Math.min(K | 0 || 1, Math.max(1, total));
    var key = new Float64Array(cap), val = new Float32Array(cap), idx = new Int32Array(cap * 4);
    var size = 0;

    function swap(x, y) {
      var t = key[x]; key[x] = key[y]; key[y] = t;
      var u = val[x]; val[x] = val[y]; val[y] = u;
      for (var z = 0; z < 4; z++) { var w = idx[x * 4 + z]; idx[x * 4 + z] = idx[y * 4 + z]; idx[y * 4 + z] = w; }
    }
    function up(c) { var p; while (c > 0) { p = (c - 1) >> 1; if (key[p] <= key[c]) return; swap(c, p); c = p; } }
    function down(c) {
      for (;;) {
        var l = 2 * c + 1, r = l + 1, s = c;
        if (l < size && key[l] < key[s]) s = l;
        if (r < size && key[r] < key[s]) s = r;
        if (s === c) return;
        swap(c, s); c = s;
      }
    }

    var i, j, a, bq, o, w, av;
    for (i = 0; i < L; i++) {
      for (j = i + 1; j < L; j++) {
        o = (i * L + j) * AA;
        for (bq = 0; bq < A; bq++) {
          for (a = 0; a < A; a++) {
            w = W[o + bq * A + a];
            av = w < 0 ? -w : w;
            if (size < cap) {
              key[size] = av; val[size] = w;
              idx[size * 4] = i; idx[size * 4 + 1] = a; idx[size * 4 + 2] = j; idx[size * 4 + 3] = bq;
              size++; up(size - 1);
            } else if (av > key[0]) {
              key[0] = av; val[0] = w;
              idx[0] = i; idx[1] = a; idx[2] = j; idx[3] = bq;
              down(0);
            }
          }
        }
      }
    }

    var out = new Float32Array(size * 5);
    for (var s2 = 0; s2 < size; s2++) {
      out[s2 * 5] = idx[s2 * 4];
      out[s2 * 5 + 1] = idx[s2 * 4 + 1];
      out[s2 * 5 + 2] = idx[s2 * 4 + 2];
      out[s2 * 5 + 3] = idx[s2 * 4 + 3];
      out[s2 * 5 + 4] = val[s2];
    }
    return out;
  };

  /** Forward pass for one sequence, plus the bias-only and coupling-only
   *  softmaxes the network panel decomposes x' into. */
  Gremlin.prototype.forward = function (sel) {
    var L = this.L, A = this.A, AA = this.AA, W = this.W, b = this.b, seqs = this.seqs;
    if (!(sel >= 0 && sel < this.N)) sel = 0;
    var logits = new Float32Array(L * A);
    var probs = new Float32Array(L * A);
    var pBias = new Float32Array(L * A);
    var pNoBias = new Float32Array(L * A);
    var noBias = new Float32Array(L * A);
    var s = sel * L, i, j, a, o, bi, rowI;

    for (i = 0; i < L; i++) {
      bi = i * A; rowI = i * L;
      for (a = 0; a < A; a++) logits[bi + a] = b[bi + a];
      for (j = 0; j < L; j++) {
        if (j === i) continue;
        o = (rowI + j) * AA + seqs[s + j] * A;
        for (a = 0; a < A; a++) logits[bi + a] += W[o + a];
      }
      for (a = 0; a < A; a++) noBias[bi + a] = logits[bi + a] - b[bi + a];
      softmaxRange(logits, probs, bi, A);
      softmaxRange(b, pBias, bi, A);
      softmaxRange(noBias, pNoBias, bi, A);
    }
    return { sel: sel, logits: logits, probs: probs, pBias: pBias, pNoBias: pNoBias };
  };

  /*
   * Swap the alignment without touching the parameters. The original recomputed
   * L and A on every keystroke and only reset the weights when the dimensions
   * actually changed; this preserves that, so editing a residue lets you watch
   * the existing model react instead of starting over.
   * L must be unchanged; N may differ.
   */
  Gremlin.prototype.setData = function (seqs, N, uniform, identity, maxRefs) {
    N = N | 0;
    if (this.wasm) {
      // The sequence buffer was sized once; a larger alignment needs a full
      // re-init. Report that rather than writing past the reservation.
      if (N * this.L > this.seqs.length) return false;
      // Copy in place and keep the full-length view: nothing reads past N*L, and
      // shrinking it would break a later, larger setData.
      this.seqs.set(seqs.subarray ? seqs.subarray(0, N * this.L) : seqs);
    } else {
      this.seqs = seqs;
    }
    this.N = N;
    if (uniform) {
      this.sw = new Float32Array(this.N);
      this.sw.fill(1);
      this.Meff = this.N;
      this.approxWeights = false;
    } else {
      this.computeWeights(identity === undefined ? 0.8 : identity, maxRefs || 3000);
    }
    if (this.sel >= this.N) this.sel = 0;
    this._allocBatch();
  };

  /*
   * Dense (L*A) x (L*A) coupling matrix in global (I, J) order, for the small
   * weights heat map on the educational page. Only ever called at toy sizes --
   * at L=128 this would be a 110MB postMessage.
   */
  Gremlin.prototype.couplingMatrix = function () {
    var L = this.L, A = this.A, AA = this.AA, W = this.W, LA = L * A;
    var out = new Float32Array(LA * LA);
    var i, j, a, bq, o;
    for (i = 0; i < L; i++) {
      for (j = 0; j < L; j++) {
        if (i === j) continue;
        o = (i * L + j) * AA;
        for (a = 0; a < A; a++) {
          for (bq = 0; bq < A; bq++) out[(i * A + a) * LA + (j * A + bq)] = W[o + bq * A + a];
        }
      }
    }
    return out;
  };

  /** The A x A coupling block for a position pair, indexed [a*A + b]. */
  Gremlin.prototype.block = function (i, j) {
    var L = this.L, A = this.A, AA = this.AA, W = this.W;
    var out = new Float32Array(AA);
    if (i === j || i < 0 || j < 0 || i >= L || j >= L) return out;
    var o = (i * L + j) * AA, a, bq;
    for (a = 0; a < A; a++) for (bq = 0; bq < A; bq++) out[a * A + bq] = W[o + bq * A + a];
    return out;
  };

  /* ------------------------------------------------------------------ */
  /* worker plumbing                                                    */
  /* ------------------------------------------------------------------ */

  /* ------------------------------------------------------------------ */
  /* backend selection                                                  */
  /* ------------------------------------------------------------------ */

  /*
   * A candidate backend has to prove itself before it is used. Both backends
   * run one step of the same small fixed problem and their gradients are
   * compared against the plain-JS reference; a backend that disagrees refuses
   * itself and the caller falls back.
   *
   * This is not ceremony. A miscompiled kernel or a wrong workgroup index would
   * otherwise produce a plausible-looking contact map that is quietly wrong,
   * which is the worst possible failure for this tool. The probe costs a few
   * milliseconds at startup.
   */
  var SELFTEST_TOL = 2e-3;
  // How much looser the parameter check is than the gradient check. See the
  // comment in compareProbe: Adam is discontinuous where the gradient vanishes.
  var W_TOL_RATIO = 25;

  function buildProbe(backend) {
    var L = 5, A = 4, N = 6, seed = 20260804;
    var st = seed >>> 0;
    var rnd = function () {
      st ^= st << 13; st >>>= 0; st ^= st >>> 17; st ^= st << 5; st >>>= 0;
      return st / 4294967296;
    };
    var seqs = new Int32Array(L * N), k;
    for (k = 0; k < L * N; k++) seqs[k] = (rnd() * A) | 0;

    var g = new Gremlin({
      L: L, A: A, N: N, seqs: seqs, uniformWeights: true, backend: backend,
      cfg: { batch: N, alpha: 0.05, beta: 0.02, lr: 0.03, regMode: 'gremlin' }
    });
    // a deterministic, non-symmetric-looking starting point
    var i, j, a, bq, c;
    st = 987654321;
    for (i = 0; i < L; i++) {
      for (j = i + 1; j < L; j++) {
        for (a = 0; a < A; a++) {
          for (bq = 0; bq < A; bq++) {
            c = (rnd() - 0.5) * 1.5;
            g.W[((i * L + j) * A + bq) * A + a] = c;
            g.W[((j * L + i) * A + a) * A + bq] = c;
          }
        }
      }
    }
    for (k = 0; k < L * A; k++) g.b[k] = (rnd() - 0.5) * 0.6;
    return g;
  }

  /*
   * Max relative deviation of a candidate backend from the JS reference.
   * Async because the GPU path has to read its results back. The probe is
   * deliberately tiny (L=5, A=4, N=6), so even a GPU round trip is milliseconds.
   */
  function selfTest(backend) {
    var ref = buildProbe(null);
    var cand = buildProbe(backend);
    // identical starting parameters (buildProbe is deterministic, but be explicit)
    if (cand.gpu) {
      cand.gpu.device.queue.writeBuffer(cand.gpu.b.W, 0, ref.W);
      cand.gpu.device.queue.writeBuffer(cand.gpu.b.bias, 0, ref.b);
      cand.gpu.dirtyW = true;
    } else {
      cand.W.set(ref.W);
      cand.b.set(ref.b);
    }

    ref.step();
    return cand.stepAsync()
      .then(function () { return cand.gpu ? cand.syncStats() : null; })
      .then(function () { return cand.gpu ? cand.gpu.readback(cand.gpu.b.G, cand.P) : cand.G; })
      .then(function (candG) {
        return (cand.gpu ? cand.gpu.readback(cand.gpu.b.Gb, cand.b.length) : Promise.resolve(cand.Gb))
          .then(function (candGb) {
            return (cand.gpu ? cand.syncParams() : Promise.resolve(true))
              .then(function () { return cand.contactMapAsync(); })
              .then(function (cmB) { return compareProbe(ref, cand, candG, candGb, cmB); });
          });
      });
  }

  function compareProbe(ref, cand, candG, candGb, cmB) {
    var worst = 0, k, d, scale;
    for (k = 0; k < ref.P; k++) {
      scale = Math.abs(ref.G[k]) + 1e-4;
      d = Math.abs(ref.G[k] - candG[k]) / scale;
      if (d > worst) worst = d;
    }
    for (k = 0; k < ref.b.length; k++) {
      scale = Math.abs(ref.Gb[k]) + 1e-4;
      d = Math.abs(ref.Gb[k] - candGb[k]) / scale;
      if (d > worst) worst = d;
    }
    /*
     * The updated parameters are checked too, but against a looser bound, and
     * deliberately so. Adam's first step is lr * g / (|g| + eps), i.e. very
     * nearly lr * sign(g) -- discontinuous at g = 0. Wherever the true gradient
     * is near zero, a difference of 1e-6 between two correct implementations can
     * flip the sign and move that parameter by a full lr. Holding W to the same
     * tolerance as the gradient would reject correct backends. A genuinely wrong
     * update is off by far more than this (the sabotage tests land at 19x).
     */
    var wWorst = 0;
    if (cand.W.length === ref.P) {
      for (k = 0; k < ref.P; k++) {
        scale = Math.abs(ref.W[k]) + 1e-4;
        d = Math.abs(ref.W[k] - cand.W[k]) / scale;
        if (d > wWorst) wWorst = d;
      }
    }
    if (wWorst / W_TOL_RATIO > worst) worst = wWorst / W_TOL_RATIO;
    var lossDev = Math.abs(ref.pll - cand.pll) / (Math.abs(ref.pll) + 1e-6);
    if (lossDev > worst) worst = lossDev;

    // a contact map too, which exercises the gauge fixing and APC
    var cmA = ref.contactMap();
    for (k = 0; k < cmA.length; k++) {
      scale = Math.abs(cmA[k]) + 1e-4;
      d = Math.abs(cmA[k] - cmB[k]) / scale;
      if (d > worst) worst = d;
    }
    return worst;
  }

  /*
   * Pick the fastest backend that passes. Order is WebGPU, then WASM, then JS.
   * JS is always available and is the reference, so selection cannot fail.
   * Returns { backend, name, tried: [{name, ok, dev, err}] }.
   */
  function selectBackend(opts) {
    opts = opts || {};
    var tried = [];
    var wasmUrl = opts.wasmUrl || 'gremlin.wasm';

    function accept(be, name) {
      var p;
      try {
        p = selfTest(be);
      } catch (e) {
        tried.push({ name: name, ok: false, err: String(e && e.message || e) });
        return Promise.resolve(null);
      }
      return p.then(function (dev) {
        var ok = dev <= SELFTEST_TOL;
        tried.push({ name: name, ok: ok, dev: dev });
        return ok ? be : null;
      }).catch(function (e) {
        tried.push({ name: name, ok: false, err: String(e && e.message || e) });
        return null;
      });
    }

    var chain = Promise.resolve(null);

    if (opts.gpu !== false && typeof GremlinGPU !== 'undefined' && GremlinGPU.available()) {
      chain = chain.then(function (got) {
        if (got) return got;
        return GremlinGPU.create().then(function (be) {
          return be ? accept(be, 'webgpu') : (tried.push({ name: 'webgpu', ok: false, err: 'unavailable' }), null);
        }).catch(function (e) {
          tried.push({ name: 'webgpu', ok: false, err: String(e && e.message || e) });
          return null;
        });
      });
    } else {
      tried.push({ name: 'webgpu', ok: false, err: 'not supported here' });
    }

    if (opts.wasm !== false) {
      chain = chain.then(function (got) {
        if (got) return got;
        return initWasm(wasmUrl).then(function (be) {
          return be ? accept(be, 'wasm') : (tried.push({ name: 'wasm', ok: false, err: 'load failed' }), null);
        });
      });
    }

    return chain.then(function (got) {
      return { backend: got, name: got ? got.name : 'js', tried: tried };
    });
  }

  var API = {
    Gremlin: Gremlin,
    softmaxRange: softmaxRange,
    initWasm: initWasm,
    selfTest: selfTest,
    selectBackend: selectBackend,
    SELFTEST_TOL: SELFTEST_TOL
  };

  if (IS_WORKER) {
    /*
     * The GPU backend is optional and lives in its own file so this one stays
     * loadable in node. A missing or broken gremlin-gpu.js just means no WebGPU.
     */
    try { self.importScripts('gremlin-gpu.js'); } catch (e) { /* no WebGPU path */ }

    var model = null;
    var backend = null;
    var backendName = 'js';
    var backendTried = [];
    var running = false;
    var scheduled = false;
    var snapEveryMs = 100;      // cap UI updates at ~10 Hz regardless of step rate
    var heavyEvery = 3;         // coupling raster + top-K are the expensive parts
    var snapCount = 0;
    var lastSnap = 0;
    // Achieved throughput between snapshots, not 1/step-duration: with pacing on,
    // raw step speed would read 33000/s while the model does 20 steps/s.
    var rateT = 0, rateS = 0, rateSps = 0;
    /*
     * Deliberate throttle. A 3x3 toy model runs at ~200k steps/s once the
     * optimizer is off the UI thread, which converges before the first frame
     * and destroys the point of watching it. Pacing is a display concern, so it
     * lives here rather than in Gremlin. 0 means unlimited.
     */
    var maxRate = 20;
    /*
     * Which heavy products the page actually displays. The coupling raster and
     * the top-K scan are each O(L^2 A^2) -- 10M reads at L=155 -- so computing
     * them for a panel that isn't on screen is pure waste. The educational page
     * wants top-K (network diagram); the practical page wants neither.
     */
    var wantCoup = false, wantTop = true;

    function post(m, xfer) { self.postMessage(m, xfer || []); }

    /*
     * Async because WebGPU has to read the contact map back off the device.
     * The JS and WASM backends resolve immediately.
     */
    async function snapshot(heavy) {
      if (!model) return;
      if (model.gpu) await model.syncStats();

      /*
       * The per-sequence forward pass, the top-K scan and the dense matrix only
       * feed panels that are illegible above a few hundred nodes, so they are
       * skipped entirely for larger models. That also means a GPU-resident model
       * never needs its parameters copied back for a snapshot -- only the L x L
       * contact map crosses, which is 96KB at L=155 against 162MB for W.
       */
      var small = model.L * model.A <= 256;

      var msg = {
        type: 'snapshot',
        steps: model.steps,
        pll: model.pll,
        regW: model.regW,
        regB: model.regB,
        loss: model.loss,
        rms: model.rms,
        L: model.L, A: model.A, N: model.N, Meff: model.Meff,
        running: running,
        sps: sps(),
        hist: model.history()
      };
      var xfer = [msg.hist.buffer];

      msg.contact = await model.contactMapAsync();
      xfer.push(msg.contact.buffer);
      msg.sel = model.sel;

      /* Everything below reads the parameters directly, so a GPU-resident model
         needs them synced first -- which syncParams only allows when small. */
      var haveParams = small && (await model.syncParams());

      if (haveParams) {
        var f = model.forward(model.sel);
        msg.logits = f.logits; msg.probs = f.probs; msg.pBias = f.pBias; msg.pNoBias = f.pNoBias;
        xfer.push(f.logits.buffer, f.probs.buffer, f.pBias.buffer, f.pNoBias.buffer);
      }

      if (heavy && haveParams) {
        if (wantCoup) {
          var ci = model.couplingImage(512);
          msg.coupImg = ci.data; msg.coupN = ci.n; msg.coupScale = ci.scale;
          xfer.push(ci.data.buffer);
        }
        if (wantTop) {
          // the network panel only draws top-K; below ~2000 that is everything
          msg.top = model.topCouplings(2000);
          xfer.push(msg.top.buffer);
        }
        msg.wmat = model.couplingMatrix();      // the educational heat map
        xfer.push(msg.wmat.buffer);
      }
      post(msg, xfer);
    }

    function sps() { return rateSps; }

    function tickRate(t) {
      if (!rateT) { rateT = t; rateS = model.steps; return; }
      var dt = t - rateT;
      if (dt >= 500) {                          // long enough to be a stable estimate
        rateSps = (model.steps - rateS) * 1000 / dt;
        rateT = t; rateS = model.steps;
      }
    }

    /*
     * Run a bounded slice of steps, then yield through the task queue so
     * pause/config messages are actually processed. The original ran on a
     * 100ms setInterval on the UI thread: once a step exceeded 100ms the
     * callbacks queued up and the page stopped responding entirely.
     */
    async function pump() {
      scheduled = false;
      if (!model || !running) return;
      var minInt = maxRate > 0 ? 1000 / maxRate : 0;
      var t0 = now(), t1;

      try {
        if (minInt > 0) {
          await model.stepAsync();
          t1 = now();
          tickRate(t1);
          // when paced slowly enough to watch, show every step
          if (maxRate <= 20 || t1 - lastSnap >= snapEveryMs) {
            lastSnap = t1;
            await snapshot((snapCount++ % heavyEvery) === 0);
          }
          schedule(Math.max(0, minInt - (now() - t0)));
          return;
        }

        var budget = 40;
        do {
          await model.stepAsync();
          t1 = now();
        } while (running && t1 - t0 < budget);

        tickRate(t1);
        /*
         * On WebGPU this snapshot is also the queue drain: stepAsync only
         * submits, so without a periodic readback the command queue would run
         * arbitrarily far ahead of the device.
         */
        if (t1 - lastSnap >= snapEveryMs) {
          lastSnap = t1;
          await snapshot((snapCount++ % heavyEvery) === 0);
        }
        schedule(0);
      } catch (err) {
        running = false;
        post({ type: 'error', message: String(err && err.message || err) });
      }
    }

    // Always go through the task queue, never a tight loop, so pause/config
    // messages are actually delivered.
    function schedule(delay) {
      if (!scheduled && running) { scheduled = true; setTimeout(pump, delay || 0); }
    }

    function now() {
      return typeof performance !== 'undefined' ? performance.now() : Date.now();
    }

    /*
     * Backend selection happens once, on the first init, and is cached. Each
     * candidate must reproduce the JS reference on a probe problem before it is
     * accepted -- see selectBackend / selfTest.
     */
    var backendReady = null;
    function ensureBackend() {
      if (!backendReady) {
        backendReady = selectBackend({}).then(function (sel) {
          backend = sel.backend;
          backendName = sel.name;
          backendTried = sel.tried;
          post({ type: 'backend', name: sel.name, tried: sel.tried });
          return sel;
        });
      }
      return backendReady;
    }

    self.onmessage = async function (ev) {
      var d = ev.data, m = d && d.type;
      try {
        if (m === 'init') {
          running = false;
          if (d.maxRate !== undefined) maxRate = d.maxRate;
          if (d.wantCoup !== undefined) wantCoup = !!d.wantCoup;
          if (d.wantTop !== undefined) wantTop = !!d.wantTop;
          post({ type: 'progress', phase: 'backend', frac: 0 });
          await ensureBackend();
          post({ type: 'progress', phase: 'weights', frac: 0 });
          model = new Gremlin({
            L: d.L, A: d.A, N: d.N, seqs: d.seqs, cfg: d.cfg,
            uniformWeights: d.uniformWeights, backend: backend,
            gap: d.gap, biasInit: d.biasInit,
            identity: d.identity, maxRefs: d.maxRefs, seed: d.seed,
            onProgress: function (f) { post({ type: 'progress', phase: 'weights', frac: f }); }
          });
          rateT = 0; rateS = 0; rateSps = 0;
          post({
            type: 'inited', L: model.L, A: model.A, N: model.N,
            Meff: model.Meff, approxWeights: model.approxWeights,
            params: model.P, bytes: model.bytes(),
            backend: backendName,
            suggestLr: Gremlin.suggestLr(model.L, model.B)
          });
          await snapshot(true);
        } else if (m === 'data') {
          // same L, new sequences: keep the learned parameters
          if (model) {
            if (model.setData(d.seqs, d.N, d.uniformWeights, d.identity, d.maxRefs) === false) {
              // more sequences than the buffers were sized for; caller must re-init
              post({ type: 'needsInit' });
            } else {
              post({
                type: 'inited', L: model.L, A: model.A, N: model.N,
                Meff: model.Meff, approxWeights: model.approxWeights,
                params: model.P, bytes: model.bytes(), backend: backendName
              });
              await snapshot(true);
            }
          }
        } else if (m === 'config') {
          if (d.maxRate !== undefined) maxRate = d.maxRate;
          if (d.wantCoup !== undefined) wantCoup = !!d.wantCoup;
          if (d.wantTop !== undefined) wantTop = !!d.wantTop;
          if (model && d.cfg) model.setConfig(d.cfg);
        } else if (m === 'run') {
          if (model && !running) { running = true; lastSnap = 0; schedule(0); }
        } else if (m === 'pause') {
          running = false;
          await snapshot(true);
        } else if (m === 'reset') {
          if (model) {
            running = false; model.reset();
            rateT = 0; rateS = 0; rateSps = 0;
            await snapshot(true);
          }
        } else if (m === 'select') {
          if (model) { model.sel = d.sel | 0; if (!running) await snapshot(false); }
        } else if (m === 'block') {
          if (model) {
            var blk = await model.blockAsync(d.i | 0, d.j | 0);
            post({ type: 'block', i: d.i | 0, j: d.j | 0, A: model.A, data: blk }, [blk.buffer]);
          }
        } else if (m === 'contacts') {
          if (model) {
            post({ type: 'contacts', list: model.topContacts(d.minSep | 0 || 5, d.limit | 0 || 0) });
          }
        } else if (m === 'snapshot') {
          await snapshot(true);
        }
      } catch (err) {
        running = false;
        post({ type: 'error', message: String(err && err.message || err), stack: String(err && err.stack || '') });
      }
    };

    post({ type: 'ready' });
  }

  root.GremlinCore = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
