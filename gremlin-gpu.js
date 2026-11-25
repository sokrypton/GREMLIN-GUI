/*
 * gremlin-gpu.js -- WebGPU backend.
 *
 * ---------------------------------------------------------------------------
 * READ THIS FIRST
 * ---------------------------------------------------------------------------
 * This file was first written where WebGPU could not be executed, and the
 * first run on real hardware (Chrome on an Apple M2, Metal) found the bugs a
 * static check and a tiny probe cannot: dispatches over the 65535-workgroup
 * limit (so at real sizes symmetrize and Adam silently never ran), a reduction
 * that overran its buffer, the bias init wiped on upload, and a coupling
 * penalty 2x the CPU paths'. test/gpu.html is the harness that found them; run
 * it after touching anything here. WebGPU errors are asynchronous -- a bad
 * dispatch just drops the command buffer -- so they are collected in
 * `errors` and any error fails the self-test.
 *
 * What makes shipping it safe anyway: gremlin-core.js will not use a backend
 * until it reproduces the JS reference's gradient, updated parameters, loss and
 * contact map on a small fixed probe problem (selfTest, 2e-3 tolerance). If any
 * kernel here is wrong, the probe disagrees, this backend refuses itself, and
 * the worker silently falls back to WASM. A wrong contact map cannot reach the
 * user; the worst case is that the GPU path is never used.
 *
 * ---------------------------------------------------------------------------
 * Design
 * ---------------------------------------------------------------------------
 * W, G, m and v stay in GPU storage buffers for the whole run -- W is 162MB at
 * L=155, so the point is that it never crosses the bus. Only the L x L contact
 * map (96KB) and a handful of scalars are read back per snapshot.
 *
 * The backward pass is the interesting one. The natural formulation scatters
 * into dW and would need f32 atomics, which WGSL does not have. Instead one
 * workgroup owns one ordered position pair (i, j) and keeps the whole A x A
 * block in registers: thread `a` holds acc[b] for all b, loops over the batch
 * accumulating acc[x_nj] += d[n][i][a], then writes A*A values once. No
 * atomics, no contention, and the same total memory traffic as the forward pass.
 *
 * Layout matches the CPU backends exactly:
 *   W[((i*L + j)*A + b)*A + a]  contributes to logit_i(a) when position j is b.
 */
(function (root) {
  'use strict';

  var MAX_A = 32;          // WGSL needs compile-time array sizes; A <= 21 here

  /* ------------------------------------------------------------------ */
  /* shaders                                                            */
  /* ------------------------------------------------------------------ */

  /*
   * Uniforms shared by every kernel, packed as 16 x 4 bytes by writeCfg. The
   * trailing pads keep it a multiple of 16 for the alignment rule; `pad` itself
   * doubles as the APC divisor, which only the apc shader reads.
   */
  var COMMON = `
struct Cfg {
  L : u32,
  A : u32,
  B : u32,
  P : u32,
  lr : f32,
  gW : f32,
  ibc1 : f32,
  ibc2 : f32,
  b1 : f32,
  b2 : f32,
  eps : f32,
  pad : f32,
  nNorm : u32,
  pad2 : u32,
  pad3 : u32,
  pad4 : u32,
};
@group(0) @binding(0) var<uniform> cfg : Cfg;
`;

  /*
   * Forward + softmax. Each workgroup takes FWD_ROWS batch items at one
   * position i: thread (a, r) handles state a of item wid.x*FWD_ROWS + r. It
   * writes the local gradient d = coef * (p - onehot) into `D`, plus the
   * per-(n,i) loss contribution into `Loss`.
   *
   * Reads are coalesced: for fixed (i, j, x_nj) the A values are contiguous and
   * consecutive threads read consecutive addresses. Packing four items per
   * workgroup (rather than one, which leaves 11 of 32 lanes idle in a tiny
   * group) was measured 1.6x faster on an M2, bit-identical output; eight was
   * no better than four. Staging the sequence in workgroup memory and
   * unrolling j with four accumulators was tried and was slower.
   */
  var FWD_ROWS = 4;
  var WGSL_FORWARD = COMMON + `
@group(0) @binding(1) var<storage, read>       W     : array<f32>;
@group(0) @binding(2) var<storage, read>       bias  : array<f32>;
@group(0) @binding(3) var<storage, read>       seqs  : array<i32>;
@group(0) @binding(4) var<storage, read>       batch : array<i32>;
@group(0) @binding(5) var<storage, read>       coef  : array<f32>;
@group(0) @binding(6) var<storage, read_write> D     : array<f32>;
@group(0) @binding(7) var<storage, read_write> Loss  : array<f32>;

var<workgroup> sh   : array<f32, ${MAX_A * FWD_ROWS}>;
var<workgroup> smax : array<f32, ${FWD_ROWS}>;
var<workgroup> ssum : array<f32, ${FWD_ROWS}>;

@compute @workgroup_size(${MAX_A}, ${FWD_ROWS})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let a = lid.x;
  let r = lid.y;
  let o = r * ${MAX_A}u;          // this item's slice of the shared arrays
  let n = wid.x * ${FWD_ROWS}u + r; // batch index
  let i = wid.y;                  // position
  // no early return: every thread must reach the barriers below
  let ok = n < cfg.B && i < L;

  var seqBase : u32 = 0u;
  if (ok) { seqBase = u32(batch[n]) * L; }
  var acc : f32 = 0.0;
  if (ok && a < A) {
    acc = bias[i * A + a];
    for (var j : u32 = 0u; j < L; j = j + 1u) {
      if (j == i) { continue; }
      let b = u32(seqs[seqBase + j]);
      acc = acc + W[(((i * L + j) * A + b) * A) + a];
    }
  }
  sh[o + a] = acc;
  workgroupBarrier();

  // stable softmax: max, then sum, each reduced by lane 0 over A <= 21 states
  if (a == 0u) {
    var mx : f32 = sh[o];
    for (var k : u32 = 1u; k < A; k = k + 1u) { mx = max(mx, sh[o + k]); }
    smax[r] = mx;
  }
  workgroupBarrier();

  var e : f32 = 0.0;
  if (a < A) { e = exp(sh[o + a] - smax[r]); }
  sh[o + a] = e;
  workgroupBarrier();

  if (a == 0u) {
    var s : f32 = 0.0;
    for (var k : u32 = 0u; k < A; k = k + 1u) { s = s + sh[o + k]; }
    ssum[r] = s;
  }
  workgroupBarrier();

  if (ok && a < A) {
    let p = sh[o + a] / ssum[r];
    let xi = u32(seqs[seqBase + i]);
    var onehot : f32 = 0.0;
    if (a == xi) { onehot = 1.0; }
    D[(n * L + i) * A + a] = coef[n] * (p - onehot);
    if (a == xi) {
      Loss[n * L + i] = -coef[n] * log(max(p, 1e-30));
    }
  }
}
`;

  /*
   * Backward. One workgroup per position i and BWD_ROWS consecutive j; thread
   * (a, r) owns pair (i, j = wid.y*BWD_ROWS + r), state a, and holds the full
   * A-vector of accumulators over b.
   * G_ij(b, a) = sum over batch items whose x_nj == b of d[n][i][a].
   * No atomics, no contention, and the same total memory traffic as forward.
   *
   * Measured alternatives on an M2, all slower or within 2%: accumulators in
   * workgroup memory (2.8x slower -- the private array is not the problem it
   * looks like), D stored position-major, and a precomputed residue table in
   * place of the batch -> seqs lookup.
   */
  var BWD_ROWS = 4;
  var WGSL_BACKWARD = COMMON + `
@group(0) @binding(1) var<storage, read>       D     : array<f32>;
@group(0) @binding(2) var<storage, read>       seqs  : array<i32>;
@group(0) @binding(3) var<storage, read>       batch : array<i32>;
@group(0) @binding(4) var<storage, read_write> G     : array<f32>;

@compute @workgroup_size(${MAX_A}, ${BWD_ROWS})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let a = lid.x;
  let i = wid.x;
  let j = wid.y * ${BWD_ROWS}u + lid.y;
  if (i >= L || j >= L) { return; }

  let blk = (i * L + j) * A * A;

  // diagonal blocks are never used; keep them zero
  if (i == j) {
    if (a < A) {
      for (var b : u32 = 0u; b < A; b = b + 1u) { G[blk + b * A + a] = 0.0; }
    }
    return;
  }
  if (a >= A) { return; }

  var acc : array<f32, ${MAX_A}>;
  for (var b : u32 = 0u; b < A; b = b + 1u) { acc[b] = 0.0; }

  for (var n : u32 = 0u; n < cfg.B; n = n + 1u) {
    let b = u32(seqs[u32(batch[n]) * L + j]);
    acc[b] = acc[b] + D[(n * L + i) * A + a];
  }

  for (var b : u32 = 0u; b < A; b = b + 1u) {
    G[blk + b * A + a] = acc[b];
  }
}
`;

  /*
   * Bias gradient: Gb[i][a] = sum over batch of d[n][i][a].
   * One thread per (i, a).
   */
  var WGSL_BIASGRAD = COMMON + `
@group(0) @binding(1) var<storage, read>       D  : array<f32>;
@group(0) @binding(2) var<storage, read_write> Gb : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let k = gid.y * nwg.x * 64u + gid.x;
  if (k >= L * A) { return; }
  let i = k / A;
  let a = k % A;
  var s : f32 = 0.0;
  for (var n : u32 = 0u; n < cfg.B; n = n + 1u) {
    s = s + D[(n * L + i) * A + a];
  }
  Gb[k] = s;
}
`;

  /*
   * Symmetrize + Adam + L2, fused, over the i<j blocks only. One thread per
   * element of the full P range; threads in the lower half and on the diagonal
   * exit at once. The symmetric total goes into both halves of G (the
   * self-test reads it), Adam runs on the i<j element, and the new value is
   * written to W in both halves -- so W stays exactly symmetric, and m, v are
   * kept for i<j only. Sq gets w*w for i<j only (its lower half stays zero), so
   * the host doubles the sum.
   */
  var WGSL_SYMADAM = COMMON + `
@group(0) @binding(1) var<storage, read_write> W  : array<f32>;
@group(0) @binding(2) var<storage, read_write> G  : array<f32>;
@group(0) @binding(3) var<storage, read_write> M  : array<f32>;
@group(0) @binding(4) var<storage, read_write> V  : array<f32>;
@group(0) @binding(5) var<storage, read_write> Sq : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let AA = A * A;
  let k = gid.y * nwg.x * 64u + gid.x;
  if (k >= cfg.P) { return; }
  let blk = k / AA;
  let i = blk / L;
  let j = blk % L;
  if (j <= i) { return; }
  let rest = k % AA;
  let b = rest / A;
  let a = rest % A;
  let oJI = (j * L + i) * AA + a * A + b;

  let s = G[k] + G[oJI];
  G[k] = s;
  G[oJI] = s;

  let w = W[k];
  let g = s + cfg.gW * w;
  let m = cfg.b1 * M[k] + (1.0 - cfg.b1) * g;
  let v = cfg.b2 * V[k] + (1.0 - cfg.b2) * g * g;
  M[k] = m;
  V[k] = v;
  let nw = w - cfg.lr * (m * cfg.ibc1) / (sqrt(v * cfg.ibc2) + cfg.eps);
  W[k] = nw;
  W[oJI] = nw;
  Sq[k] = w * w;
}
`;

  /* Same for the bias vector, with its own penalty coefficient in gW. */
  var WGSL_ADAMB = COMMON + `
@group(0) @binding(1) var<storage, read_write> b  : array<f32>;
@group(0) @binding(2) var<storage, read>       Gb : array<f32>;
@group(0) @binding(3) var<storage, read_write> Mb : array<f32>;
@group(0) @binding(4) var<storage, read_write> Vb : array<f32>;
@group(0) @binding(5) var<storage, read_write> Sq : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
  let k = gid.y * nwg.x * 64u + gid.x;
  if (k >= cfg.P) { return; }
  let w = b[k];
  let g = Gb[k] + cfg.gW * w;
  let m = cfg.b1 * Mb[k] + (1.0 - cfg.b1) * g;
  let v = cfg.b2 * Vb[k] + (1.0 - cfg.b2) * g * g;
  Mb[k] = m;
  Vb[k] = v;
  b[k] = w - cfg.lr * (m * cfg.ibc1) / (sqrt(v * cfg.ibc2) + cfg.eps);
  Sq[k] = w * w;
}
`;

  /*
   * Contact map: zero-sum gauge per A x A block, then Frobenius norm. One
   * workgroup per (i, j) with i < j, one thread per row of the block. The
   * average product correction needs only the L row sums of the result, so it
   * runs on the host after the one readback.
   */
  var WGSL_FROB = COMMON + `
@group(0) @binding(1) var<storage, read>       W : array<f32>;
@group(0) @binding(2) var<storage, read_write> F : array<f32>;

var<workgroup> rsum : array<f32, ${MAX_A}>;   // per row b: sum over a
var<workgroup> csum : array<f32, ${MAX_A}>;   // per column a: sum over b
var<workgroup> part : array<f32, ${MAX_A}>;

@compute @workgroup_size(${MAX_A})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let i = wid.x;
  let j = wid.y;
  let t = lid.x;
  if (i >= L || j >= L) { return; }
  if (j <= i) {
    if (i == j && t == 0u) { F[i * L + i] = 0.0; }
    return;
  }
  let blk = (i * L + j) * A * A;
  // nNorm excludes the gap state; blocks are still strided by the full A
  let nA = cfg.nNorm;

  var r : f32 = 0.0;
  var c : f32 = 0.0;
  if (t < nA) {
    for (var q : u32 = 0u; q < nA; q = q + 1u) {
      r = r + W[blk + t * A + q];
      c = c + W[blk + q * A + t];
    }
  }
  rsum[t] = r;
  csum[t] = c;
  workgroupBarrier();

  let invA = 1.0 / f32(nA);
  var tot : f32 = 0.0;
  for (var q : u32 = 0u; q < nA; q = q + 1u) { tot = tot + rsum[q]; }
  let all = tot / f32(nA * nA);

  var s : f32 = 0.0;
  if (t < nA) {
    let off = all - rsum[t] * invA;
    for (var q : u32 = 0u; q < nA; q = q + 1u) {
      let w = W[blk + t * A + q] - csum[q] * invA + off;
      s = s + w * w;
    }
  }
  part[t] = s;
  workgroupBarrier();

  if (t == 0u) {
    var f : f32 = 0.0;
    for (var q : u32 = 0u; q < ${MAX_A}u; q = q + 1u) { f = f + part[q]; }
    f = sqrt(f);
    F[i * L + j] = f;
    F[j * L + i] = f;
  }
}
`;

  /*
   * Strided sum into a fixed number of partials, so the host reads back
   * REDUCE_WG floats instead of P. Used for sum(w^2) (the reported L2 penalty)
   * and for the per-(n,i) loss terms. `P` in cfg is the element count.
   */
  var REDUCE_WG = 256;

  var WGSL_REDUCE = COMMON + `
@group(0) @binding(1) var<storage, read>       src  : array<f32>;
@group(0) @binding(2) var<storage, read_write> part : array<f32>;

var<workgroup> acc : array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let stride = 64u * ${REDUCE_WG}u;
  var s : f32 = 0.0;
  var k : u32 = wid.x * 64u + lid.x;
  loop {
    if (k >= cfg.P) { break; }
    s = s + src[k];
    k = k + stride;
  }
  acc[lid.x] = s;
  workgroupBarrier();
  if (lid.x == 0u) {
    var t : f32 = 0.0;
    for (var q : u32 = 0u; q < 64u; q = q + 1u) { t = t + acc[q]; }
    part[wid.x] = t;
  }
}
`;

  root.GremlinGPUShaders = {
    forward: WGSL_FORWARD, backward: WGSL_BACKWARD, biasgrad: WGSL_BIASGRAD,
    symadam: WGSL_SYMADAM, adamb: WGSL_ADAMB,
    frob: WGSL_FROB, reduce: WGSL_REDUCE,
    MAX_A: MAX_A, REDUCE_WG: REDUCE_WG
  };

  /* ------------------------------------------------------------------ */
  /* backend                                                            */
  /* ------------------------------------------------------------------ */

  function available() {
    return typeof navigator !== 'undefined' && !!navigator.gpu;
  }

  function GpuBackend(device, pipes) {
    this.name = 'webgpu';
    this.device = device;
    this.pipes = pipes;
    this.buf = null;
    /*
     * WebGPU reports most failures asynchronously: an invalid dispatch or copy
     * does not throw, the command buffer is just dropped. Unwatched, that looks
     * like a model that silently stops training. Collect them so the self-test
     * and the worker can refuse to go on.
     */
    var errs = this.errors = [];
    device.addEventListener('uncapturederror', function (e) {
      if (errs.length < 20) errs.push(String(e.error && e.error.message || e.error));
    });
    var self = this;
    device.lost.then(function (info) { errs.push('device lost: ' + info.message); self.lost = true; });
  }

  /* Workgroups for n threads of 64, folded into 2D past the 65535-per-dimension
     limit. Kernels rebuild the flat index from num_workgroups. */
  function grid(n) {
    var wg = Math.ceil(n / 64), x = Math.min(wg, 65535);
    return [x, Math.ceil(wg / x)];
  }

  /*
   * Compile every pipeline up front. A WGSL error surfaces here as a rejected
   * promise, which selectBackend treats as "unavailable" and falls back.
   */
  function create() {
    if (!available()) return Promise.resolve(null);
    return navigator.gpu.requestAdapter().then(function (adapter) {
      if (!adapter) return null;
      // Ask for enough headroom that L up to ~256 fits in one buffer.
      var want = {};
      var lim = adapter.limits;
      if (lim) {
        want.maxStorageBufferBindingSize = lim.maxStorageBufferBindingSize;
        want.maxBufferSize = lim.maxBufferSize;
        want.maxComputeInvocationsPerWorkgroup = lim.maxComputeInvocationsPerWorkgroup;
      }
      return adapter.requestDevice({ requiredLimits: want }).then(function (device) {
        var S = root.GremlinGPUShaders;
        var names = ['forward', 'backward', 'biasgrad', 'symadam', 'adamb', 'frob', 'reduce'];
        var pipes = {};
        device.pushErrorScope('validation');
        names.forEach(function (n) {
          pipes[n] = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: S[n] }), entryPoint: 'main' }
          });
        });
        return device.popErrorScope().then(function (err) {
          if (err) throw new Error('WGSL validation: ' + err.message);
          return new GpuBackend(device, pipes);
        });
      });
    }).catch(function (e) { root.GremlinGPU.lastError = String(e && e.message || e); return null; });
  }

  /* ------------------------------------------------------------------ */
  /*
   * Allocation. The GPU backend cannot hand JS a live view of W the way the
   * WASM backend can, so gremlin-core.js keeps its own CPU copies for the
   * read-only helpers (topCouplings, couplingMatrix, block, forward) and this
   * backend syncs them on demand. Those helpers only run at toy sizes or on a
   * single A x A block, so the sync is bounded.
   */
  GpuBackend.prototype.setup = function (g) {
    var d = this.device;
    var P = g.P, LA = g.L * g.A, Bmax = g.Bmax;
    var U = GPUBufferUsage;
    var mk = function (n, extra) {
      return d.createBuffer({ size: Math.max(4, n * 4), usage: U.STORAGE | U.COPY_DST | U.COPY_SRC | (extra || 0) });
    };
    this.b = {
      W: mk(P), G: mk(P), M: mk(P), V: mk(P), Sq: mk(P),
      bias: mk(LA), Gb: mk(LA), Mb: mk(LA), Vb: mk(LA), SqB: mk(LA),
      seqs: mk(g.N * g.L), batch: mk(Bmax), coef: mk(Bmax),
      D: mk(Bmax * g.L * g.A), Loss: mk(Bmax * g.L),
      F: mk(g.L * g.L),
      part: mk(root.GremlinGPUShaders.REDUCE_WG),
      cfg: d.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST })
    };
    // staging for readbacks; bigger ones are copied through it in chunks
    this.readN = Math.max(g.L * g.L, Math.min(P, this.SYNC_MAX), root.GremlinGPUShaders.REDUCE_WG);
    this.read = d.createBuffer({ size: this.readN * 4, usage: U.MAP_READ | U.COPY_DST });
    d.queue.writeBuffer(this.b.seqs, 0, g.seqs);
    this.zeroAll(g);
    // the starting point is whatever the CPU side set up (e.g. the frequency
    // bias init), not zeros
    // (large models drop the CPU copy of W; it starts at zero, as zeroAll left it)
    if (g.W.length >= P) d.queue.writeBuffer(this.b.W, 0, g.W, 0, P);
    d.queue.writeBuffer(this.b.bias, 0, g.b, 0, LA);
    this.dirtyW = false;
  };

  GpuBackend.prototype.zeroAll = function (g) {
    var d = this.device, z = new Float32Array(Math.max(g.P, g.L * g.A));
    ['W', 'M', 'V'].forEach(function (k) { d.queue.writeBuffer(this.b[k], 0, z, 0, g.P); }, this);
    ['bias', 'Mb', 'Vb'].forEach(function (k) { d.queue.writeBuffer(this.b[k], 0, z, 0, g.L * g.A); }, this);
  };

  GpuBackend.prototype.writeCfg = function (g, extra) {
    var a = new ArrayBuffer(64);
    var u = new Uint32Array(a), f = new Float32Array(a);
    u[0] = g.L; u[1] = g.A; u[2] = g.B; u[3] = extra.P >>> 0;
    f[4] = extra.lr || 0; f[5] = extra.gW || 0;
    f[6] = extra.ibc1 || 0; f[7] = extra.ibc2 || 0;
    f[8] = g.cfg.b1; f[9] = g.cfg.b2; f[10] = g.cfg.eps; f[11] = extra.pad || 0;
    u[12] = g.normA >>> 0;
    this.device.queue.writeBuffer(this.b.cfg, 0, a);
  };

  GpuBackend.prototype.bind = function (pipe, buffers) {
    var entries = [{ binding: 0, resource: { buffer: this.b.cfg } }];
    for (var i = 0; i < buffers.length; i++) {
      entries.push({ binding: i + 1, resource: { buffer: buffers[i] } });
    }
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0), entries: entries
    });
  };

  GpuBackend.prototype.pass = function (enc, name, buffers, x, y) {
    var p = this.pipes[name];
    var c = enc.beginComputePass();
    c.setPipeline(p);
    c.setBindGroup(0, this.bind(p, buffers));
    c.dispatchWorkgroups(x, y || 1);
    c.end();
  };

  /*
   * Run one read operation after every earlier one has finished.
   *
   * The worker's message handler and its training loop interleave at every
   * await, so a click (a coupling block, a contact list) can arrive while a
   * snapshot is mid-readback. Two overlapping readbacks map the one staging
   * buffer twice -- "Buffer already has an outstanding map pending" -- and,
   * worse, sum() and contactMap() stage their results in shared scratch
   * buffers (`part`, `F`) before reading them, so an interleaved second call
   * could overwrite the first's result before it is copied out. So the unit
   * that is serialized is the whole operation, not just the map.
   */
  GpuBackend.prototype._serial = function (fn) {
    var p = (this.q || Promise.resolve()).then(fn);
    this.q = p.catch(function () {});        // a failed read must not wedge the queue
    return p;
  };

  /* Copy `n` floats from a GPU buffer. */
  GpuBackend.prototype.readback = function (buf, n, off) {
    var self = this;
    return this._serial(function () { return self._readback(buf, n, off); });
  };

  /* Unserialized; only for use inside a _serial operation. */
  GpuBackend.prototype._readback = function (buf, n, off) {
    var d = this.device, rd = this.read, rn = this.readN, out = new Float32Array(n);
    off = off || 0;
    function chunk(done) {
      if (done >= n) return Promise.resolve(out);
      var c = Math.min(n - done, rn);
      var enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buf, (off + done) * 4, rd, 0, c * 4);
      d.queue.submit([enc.finish()]);
      return rd.mapAsync(GPUMapMode.READ, 0, c * 4).then(function () {
        out.set(new Float32Array(rd.getMappedRange(0, c * 4)), done);
        rd.unmap();
        return chunk(done + c);
      });
    }
    return chunk(0);
  };

  /*
   * One optimizer step, entirely on device.
   *
   * Encoding and submission do not block. The returned promise resolves when
   * the previous step has finished on the device, which keeps exactly one step
   * queued behind the one running -- enough to hide the encode cost, not enough
   * to let the host run away.
   *
   * Loss and the L2 penalty are reduced on device and read back lazily by
   * syncScalars(), so a caller that only wants the map never pays for them.
   */
  GpuBackend.prototype.step = function (g, adamParams) {
    var d = this.device, b = this.b;
    var L = g.L, A = g.A, B = g.B, P = g.P, LA = L * A;

    d.queue.writeBuffer(b.batch, 0, g.batch, 0, B);
    d.queue.writeBuffer(b.coef, 0, g.coef, 0, B);

    var enc = d.createCommandEncoder();

    this.writeCfg(g, { P: P, lr: adamParams.lr, gW: adamParams.gW,
                       ibc1: adamParams.ibc1, ibc2: adamParams.ibc2 });

    // forward + softmax + local gradient: FWD_ROWS batch items per workgroup
    this.pass(enc, 'forward', [b.W, b.bias, b.seqs, b.batch, b.coef, b.D, b.Loss],
              Math.ceil(B / FWD_ROWS), L);
    // couplings gradient: one thread row per ordered position pair, no atomics
    this.pass(enc, 'backward', [b.D, b.seqs, b.batch, b.G], L, Math.ceil(L / BWD_ROWS));
    var gl = grid(LA), gp = grid(P);
    this.pass(enc, 'biasgrad', [b.D, b.Gb], gl[0], gl[1]);
    this.pass(enc, 'symadam', [b.W, b.G, b.M, b.V, b.Sq], gp[0], gp[1]);

    // bias Adam uses its own penalty coefficient, so re-write cfg mid-encode
    d.queue.submit([enc.finish()]);
    this.writeCfg(g, { P: LA, lr: adamParams.lr, gW: adamParams.gWb,
                       ibc1: adamParams.ibc1, ibc2: adamParams.ibc2 });
    var enc2 = d.createCommandEncoder();
    this.pass(enc2, 'adamb', [b.bias, b.Gb, b.Mb, b.Vb, b.SqB], gl[0], gl[1]);
    d.queue.submit([enc2.finish()]);

    this.dirtyW = true;
    /*
     * Let the host run at most one step ahead of the device. Submitting never
     * blocks, so the worker's 40ms slice would otherwise queue hundreds of
     * steps -- seconds of GPU work -- and every snapshot's readback would wait
     * for all of it: a frozen display and a wildly overstated steps/s.
     */
    var prev = this.inflight;
    this.inflight = d.queue.onSubmittedWorkDone();
    return prev || Promise.resolve();
  };

  /** Reduce a device array to a scalar via partials. */
  GpuBackend.prototype.sum = function (g, src, n) {
    var S = root.GremlinGPUShaders;
    this.writeCfg(g, { P: n });
    var enc = this.device.createCommandEncoder();
    this.pass(enc, 'reduce', [src, this.b.part], S.REDUCE_WG);
    this.device.queue.submit([enc.finish()]);
    return this._readback(this.b.part, S.REDUCE_WG).then(function (p) {
      var s = 0;
      for (var i = 0; i < p.length; i++) s += p[i];
      return s;
    });
  };

  /** -PLL and sum(w^2), sum(b^2) for the reported loss breakdown. */
  GpuBackend.prototype.syncScalars = function (g) {
    var self = this;
    return this._serial(function () { return self._syncScalars(g); });
  };
  GpuBackend.prototype._syncScalars = function (g) {
    var self = this, out = {};
    return this.sum(g, this.b.Loss, g.B * g.L).then(function (pll) {
      out.pll = pll;
      return self.sum(g, self.b.Sq, g.P);
    }).then(function (sw2) {
      out.sw2 = 2 * sw2;               // Sq holds the i<j half only
      return self.sum(g, self.b.SqB, g.L * g.A);
    }).then(function (sb2) {
      out.sb2 = sb2;
      return out;
    });
  };

  /*
   * Contact map: gauge-fix + Frobenius norm on device, one readback of the
   * L x L norms (96KB at L=155), then APC on the host -- it needs only their
   * row sums, and doing it here saves a pass and a second round trip. The
   * readback is also this step's queue drain point.
   */
  GpuBackend.prototype.contactMap = function (g) {
    var self = this;
    return this._serial(function () { return self._contactMap(g); });
  };
  GpuBackend.prototype._contactMap = function (g) {
    var b = this.b, L = g.L;
    this.writeCfg(g, { P: L * L });
    var enc = this.device.createCommandEncoder();
    this.pass(enc, 'frob', [b.W, b.F], L, L);
    this.device.queue.submit([enc.finish()]);

    return this._readback(b.F, L * L).then(function (F) {
      // same arithmetic as the CPU paths: f32 row sums, f64 total
      var rs = new Float32Array(L), tot = 0, i, j, r;
      for (i = 0; i < L; i++) {
        r = 0;
        for (j = 0; j < L; j++) r = Math.fround(r + F[i * L + j]);
        rs[i] = r; tot += r;
      }
      var inv = Math.fround(1 / (tot + 1e-8));
      for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
          F[i * L + j] = i === j ? 0 : F[i * L + j] - rs[i] * rs[j] * inv;
        }
      }
      return F;
    });
  };

  /*
   * Pull W back into the CPU shadow so the read-only helpers (topCouplings,
   * couplingMatrix, forward, block) work. P floats is 42MB at L=155, so this is
   * only allowed for small models -- the callers are all gated on L*A <= 256
   * anyway, since none of those views is legible above that.
   */
  GpuBackend.prototype.SYNC_MAX = 1 << 20;   // 1M floats = 4MB

  GpuBackend.prototype.syncW = function (g) {
    if (g.P > this.SYNC_MAX) return Promise.resolve(false);
    var self = this;
    return this._serial(function () {
      if (!self.dirtyW) return true;
      return self._readback(self.b.W, g.P).then(function (w) {
        g.W.set(w);
        return self._readback(self.b.bias, g.L * g.A);
      }).then(function (bb) {
        g.b.set(bb);
        self.dirtyW = false;
        return true;
      });
    });
  };

  /** One A x A coupling block, for the inspector. Small enough to always allow. */
  GpuBackend.prototype.blockAt = function (g, i, j) {
    return this.readback(this.b.W, g.AA, (i * g.L + j) * g.AA);
  };

  GpuBackend.prototype.reset = function (g) { this.zeroAll(g); this.dirtyW = true; };

  /* Free the device buffers; the worker builds one model at a time. */
  GpuBackend.prototype.release = function () {
    this.inflight = null;
    if (!this.b) return;
    /* A snapshot may still be reading the old model's buffers; destroy them
       once the read queue has drained, not under it. */
    var old = this.b, oldRead = this.read;
    this.b = null; this.read = null;
    (this.q || Promise.resolve()).then(function () {
      for (var k in old) old[k].destroy();
      if (oldRead) oldRead.destroy();
    });
    this.q = null;
  };

  root.GremlinGPU = { available: available, create: create, GpuBackend: GpuBackend };
})(typeof globalThis !== 'undefined' ? globalThis : this);
