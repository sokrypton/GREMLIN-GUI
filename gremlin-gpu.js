/*
 * gremlin-gpu.js -- WebGPU backend.
 *
 * ---------------------------------------------------------------------------
 * READ THIS FIRST
 * ---------------------------------------------------------------------------
 * This file was written in an environment where WebGPU could not be executed
 * (Playwright's Chromium ships with the API disabled, and there is no Vulkan
 * driver), so unlike the JS and WASM paths it has never been run. The WGSL is
 * statically validated with naga in test/wgsl.test.mjs -- that catches syntax,
 * type and binding errors but not a wrong index expression.
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
   * Uniforms shared by every kernel. `pad` keeps the struct at 32 bytes, which
   * satisfies the 16-byte alignment rule with room to spare.
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
};
@group(0) @binding(0) var<uniform> cfg : Cfg;
`;

  /*
   * Forward + softmax. One workgroup per (batch item, position i), one thread
   * per state a. Writes the local gradient d = coef * (p - onehot) into `D`,
   * plus the per-(n,i) loss contribution into `Loss`.
   *
   * Reads are coalesced: for fixed (i, j, x_nj) the A values are contiguous and
   * consecutive threads read consecutive addresses.
   */
  var WGSL_FORWARD = COMMON + `
@group(0) @binding(1) var<storage, read>       W     : array<f32>;
@group(0) @binding(2) var<storage, read>       bias  : array<f32>;
@group(0) @binding(3) var<storage, read>       seqs  : array<i32>;
@group(0) @binding(4) var<storage, read>       batch : array<i32>;
@group(0) @binding(5) var<storage, read>       coef  : array<f32>;
@group(0) @binding(6) var<storage, read_write> D     : array<f32>;
@group(0) @binding(7) var<storage, read_write> Loss  : array<f32>;

var<workgroup> sh   : array<f32, ${MAX_A}>;
var<workgroup> smax : f32;
var<workgroup> ssum : f32;

@compute @workgroup_size(${MAX_A})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let a = lid.x;
  let n = wid.x;                 // batch index
  let i = wid.y;                 // position
  if (n >= cfg.B || i >= L) { return; }

  let seqBase = u32(batch[n]) * L;
  var acc : f32 = 0.0;
  if (a < A) {
    acc = bias[i * A + a];
    for (var j : u32 = 0u; j < L; j = j + 1u) {
      if (j == i) { continue; }
      let b = u32(seqs[seqBase + j]);
      acc = acc + W[(((i * L + j) * A + b) * A) + a];
    }
  }
  sh[a] = acc;
  workgroupBarrier();

  // stable softmax: max, then sum, both reduced by thread 0 over A <= 21 lanes
  if (a == 0u) {
    var mx : f32 = sh[0];
    for (var k : u32 = 1u; k < A; k = k + 1u) { mx = max(mx, sh[k]); }
    smax = mx;
  }
  workgroupBarrier();

  var e : f32 = 0.0;
  if (a < A) { e = exp(sh[a] - smax); }
  sh[a] = e;
  workgroupBarrier();

  if (a == 0u) {
    var s : f32 = 0.0;
    for (var k : u32 = 0u; k < A; k = k + 1u) { s = s + sh[k]; }
    ssum = s;
  }
  workgroupBarrier();

  if (a < A) {
    let p = sh[a] / ssum;
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
   * Backward. One workgroup per ordered pair (i, j), i != j; one thread per
   * state a; each thread holds the full A-vector of accumulators over b.
   * G_ij(b, a) = sum over batch items whose x_nj == b of d[n][i][a].
   */
  var WGSL_BACKWARD = COMMON + `
@group(0) @binding(1) var<storage, read>       D     : array<f32>;
@group(0) @binding(2) var<storage, read>       seqs  : array<i32>;
@group(0) @binding(3) var<storage, read>       batch : array<i32>;
@group(0) @binding(4) var<storage, read_write> G     : array<f32>;

@compute @workgroup_size(${MAX_A})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let a = lid.x;
  let i = wid.x;
  let j = wid.y;
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
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let k = gid.x;
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
   * Symmetrize: total_ij(b,a) = G_ij(b,a) + G_ji(a,b), written to both halves.
   * One thread per (i<j, b, a) so each pair is visited once and both writes come
   * from the same invocation -- no read-after-write hazard between threads.
   */
  var WGSL_SYM = COMMON + `
@group(0) @binding(1) var<storage, read_write> G : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let AA = A * A;
  let k = gid.x;
  let total = L * L * AA;
  if (k >= total) { return; }
  let blk = k / AA;
  let i = blk / L;
  let j = blk % L;
  if (j <= i) { return; }                 // upper triangle drives both writes
  let rest = k % AA;
  let b = rest / A;
  let a = rest % A;
  let oIJ = (i * L + j) * AA + b * A + a;
  let oJI = (j * L + i) * AA + a * A + b;
  let s = G[oIJ] + G[oJI];
  G[oIJ] = s;
  G[oJI] = s;
}
`;

  /*
   * Fused Adam + L2. Also writes w*w into `Sq` so the host can reduce it for the
   * reported penalty; a second pass reduces Sq.
   */
  var WGSL_ADAM = COMMON + `
@group(0) @binding(1) var<storage, read_write> W  : array<f32>;
@group(0) @binding(2) var<storage, read>       G  : array<f32>;
@group(0) @binding(3) var<storage, read_write> M  : array<f32>;
@group(0) @binding(4) var<storage, read_write> V  : array<f32>;
@group(0) @binding(5) var<storage, read_write> Sq : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  if (k >= cfg.P) { return; }
  let w = W[k];
  let g = G[k] + cfg.gW * w;
  let m = cfg.b1 * M[k] + (1.0 - cfg.b1) * g;
  let v = cfg.b2 * V[k] + (1.0 - cfg.b2) * g * g;
  M[k] = m;
  V[k] = v;
  W[k] = w - cfg.lr * (m * cfg.ibc1) / (sqrt(v * cfg.ibc2) + cfg.eps);
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
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
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
   * Contact map, part 1: zero-sum gauge per A x A block then Frobenius norm.
   * One workgroup per (i, j) with i < j.
   */
  var WGSL_FROB = COMMON + `
@group(0) @binding(1) var<storage, read>       W : array<f32>;
@group(0) @binding(2) var<storage, read_write> F : array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid : vec3<u32>) {
  let L = cfg.L;
  let A = cfg.A;
  let AA = A * A;
  let i = wid.x;
  let j = wid.y;
  if (i >= L || j >= L) { return; }
  if (j <= i) {
    if (i == j) { F[i * L + i] = 0.0; }
    return;
  }
  let blk = (i * L + j) * AA;

  var rowM : array<f32, ${MAX_A}>;
  var colM : array<f32, ${MAX_A}>;
  for (var k : u32 = 0u; k < A; k = k + 1u) { rowM[k] = 0.0; colM[k] = 0.0; }
  var all : f32 = 0.0;
  for (var b : u32 = 0u; b < A; b = b + 1u) {
    for (var a : u32 = 0u; a < A; a = a + 1u) {
      let w = W[blk + b * A + a];
      rowM[a] = rowM[a] + w;
      colM[b] = colM[b] + w;
      all = all + w;
    }
  }
  let invA = 1.0 / f32(A);
  for (var k : u32 = 0u; k < A; k = k + 1u) { rowM[k] = rowM[k] * invA; colM[k] = colM[k] * invA; }
  all = all / f32(AA);

  var s : f32 = 0.0;
  for (var b : u32 = 0u; b < A; b = b + 1u) {
    for (var a : u32 = 0u; a < A; a = a + 1u) {
      let w = W[blk + b * A + a] - rowM[a] - colM[b] + all;
      s = s + w * w;
    }
  }
  let f = sqrt(s);
  F[i * L + j] = f;
  F[j * L + i] = f;
}
`;

  /*
   * Contact map, part 2: average product correction. Row sums and the grand
   * total are computed on the host (L values, negligible) and passed back in.
   */
  var WGSL_APC = COMMON + `
@group(0) @binding(1) var<storage, read>       F   : array<f32>;
@group(0) @binding(2) var<storage, read>       rs  : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let L = cfg.L;
  let k = gid.x;
  if (k >= L * L) { return; }
  let i = k / L;
  let j = k % L;
  if (i == j) { out[k] = 0.0; return; }
  out[k] = F[k] - rs[i] * rs[j] * cfg.pad;   // pad carries 1/(total + 1e-8)
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
    sym: WGSL_SYM, adam: WGSL_ADAM, adamb: WGSL_ADAMB,
    frob: WGSL_FROB, apc: WGSL_APC, reduce: WGSL_REDUCE,
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
        var names = ['forward', 'backward', 'biasgrad', 'sym', 'adam', 'adamb', 'frob', 'apc', 'reduce'];
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
    }).catch(function () { return null; });
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
      F: mk(g.L * g.L), rs: mk(g.L), out: mk(g.L * g.L),
      cfg: d.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST })
    };
    this.read = d.createBuffer({ size: Math.max(g.L * g.L, P) * 4, usage: U.MAP_READ | U.COPY_DST });
    d.queue.writeBuffer(this.b.seqs, 0, g.seqs);
    this.zeroAll(g);
  };

  GpuBackend.prototype.zeroAll = function (g) {
    var d = this.device, z = new Float32Array(Math.max(g.P, g.L * g.A));
    ['W', 'M', 'V'].forEach(function (k) { d.queue.writeBuffer(this.b[k], 0, z, 0, g.P); }, this);
    ['bias', 'Mb', 'Vb'].forEach(function (k) { d.queue.writeBuffer(this.b[k], 0, z, 0, g.L * g.A); }, this);
  };

  GpuBackend.prototype.writeCfg = function (g, extra) {
    var a = new ArrayBuffer(48);
    var u = new Uint32Array(a), f = new Float32Array(a);
    u[0] = g.L; u[1] = g.A; u[2] = g.B; u[3] = extra.P >>> 0;
    f[4] = extra.lr || 0; f[5] = extra.gW || 0;
    f[6] = extra.ibc1 || 0; f[7] = extra.ibc2 || 0;
    f[8] = g.cfg.b1; f[9] = g.cfg.b2; f[10] = g.cfg.eps; f[11] = extra.pad || 0;
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

  /* Copy `n` floats from a GPU buffer. Serialized on `read`, so callers must
     not overlap readbacks. */
  GpuBackend.prototype.readback = function (buf, n) {
    var d = this.device, self = this;
    var enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, this.read, 0, n * 4);
    d.queue.submit([enc.finish()]);
    return this.read.mapAsync(GPUMapMode.READ, 0, n * 4).then(function () {
      var out = new Float32Array(self.read.getMappedRange(0, n * 4).slice(0));
      self.read.unmap();
      return out;
    });
  };

  /*
   * One optimizer step, entirely on device.
   *
   * Encoding and submission do not block, so this returns as soon as the work is
   * queued. Nothing here reads back -- the queue is drained by the periodic
   * contactMap() readback in the snapshot path, which doubles as the throttle
   * that stops submissions running away ahead of the GPU.
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

    // forward + softmax + local gradient: one workgroup per (batch item, position)
    this.pass(enc, 'forward', [b.W, b.bias, b.seqs, b.batch, b.coef, b.D, b.Loss], B, L);
    // couplings gradient: one workgroup per ordered position pair, no atomics
    this.pass(enc, 'backward', [b.D, b.seqs, b.batch, b.G], L, L);
    this.pass(enc, 'biasgrad', [b.D, b.Gb], Math.ceil(LA / 64));
    this.pass(enc, 'sym', [b.G], Math.ceil(P / 64));
    this.pass(enc, 'adam', [b.W, b.G, b.M, b.V, b.Sq], Math.ceil(P / 64));

    // bias Adam uses its own penalty coefficient, so re-write cfg mid-encode
    d.queue.submit([enc.finish()]);
    this.writeCfg(g, { P: LA, lr: adamParams.lr, gW: adamParams.gWb,
                       ibc1: adamParams.ibc1, ibc2: adamParams.ibc2 });
    var enc2 = d.createCommandEncoder();
    this.pass(enc2, 'adamb', [b.bias, b.Gb, b.Mb, b.Vb, b.SqB], Math.ceil(LA / 64));
    d.queue.submit([enc2.finish()]);

    this.dirtyW = true;
    return Promise.resolve();
  };

  /** Reduce a device array to a scalar via partials. */
  GpuBackend.prototype.sum = function (g, src, n) {
    var S = root.GremlinGPUShaders;
    this.writeCfg(g, { P: n });
    var enc = this.device.createCommandEncoder();
    this.pass(enc, 'reduce', [src, this.b.F], S.REDUCE_WG);   // F reused as partials
    this.device.queue.submit([enc.finish()]);
    return this.readback(this.b.F, S.REDUCE_WG).then(function (p) {
      var s = 0;
      for (var i = 0; i < p.length; i++) s += p[i];
      return s;
    });
  };

  /** -PLL and sum(w^2), sum(b^2) for the reported loss breakdown. */
  GpuBackend.prototype.syncScalars = function (g) {
    var self = this, out = {};
    return this.sum(g, this.b.Loss, g.B * g.L).then(function (pll) {
      out.pll = pll;
      return self.sum(g, self.b.Sq, g.P);
    }).then(function (sw2) {
      out.sw2 = sw2;
      return self.sum(g, self.b.SqB, g.L * g.A);
    }).then(function (sb2) {
      out.sb2 = sb2;
      return out;
    });
  };

  /*
   * Contact map on device: gauge-fix + Frobenius norm per block, then APC. The
   * row sums for APC come back to the host (L floats) between the two passes,
   * which is also this step's queue drain point.
   */
  GpuBackend.prototype.contactMap = function (g) {
    var self = this, b = this.b, L = g.L;
    this.writeCfg(g, { P: L * L });
    var enc = this.device.createCommandEncoder();
    this.pass(enc, 'frob', [b.W, b.F], L, L);
    this.device.queue.submit([enc.finish()]);

    return this.readback(b.F, L * L).then(function (F) {
      var rs = new Float32Array(L), tot = 0, i, j, r;
      for (i = 0; i < L; i++) {
        r = 0;
        for (j = 0; j < L; j++) r += F[i * L + j];
        rs[i] = r; tot += r;
      }
      self.device.queue.writeBuffer(b.rs, 0, rs);
      self.writeCfg(g, { P: L * L, pad: 1 / (tot + 1e-8) });
      var e2 = self.device.createCommandEncoder();
      self.pass(e2, 'apc', [b.F, b.rs, b.out], Math.ceil(L * L / 64));
      self.device.queue.submit([e2.finish()]);
      return self.readback(b.out, L * L);
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
    if (!this.dirtyW) return Promise.resolve(true);
    var self = this;
    return this.readback(this.b.W, g.P).then(function (w) {
      g.W.set(w);
      return self.readback(self.b.bias, g.L * g.A);
    }).then(function (bb) {
      g.b.set(bb);
      self.dirtyW = false;
      return true;
    });
  };

  /** One A x A coupling block, for the inspector. Small enough to always allow. */
  GpuBackend.prototype.blockAt = function (g, i, j) {
    var AA = g.AA, off = (i * g.L + j) * AA;
    var d = this.device, self = this;
    var enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.b.W, off * 4, this.read, 0, AA * 4);
    d.queue.submit([enc.finish()]);
    return this.read.mapAsync(GPUMapMode.READ, 0, AA * 4).then(function () {
      var out = new Float32Array(self.read.getMappedRange(0, AA * 4).slice(0));
      self.read.unmap();
      return out;
    });
  };

  GpuBackend.prototype.reset = function (g) { this.zeroAll(g); this.dirtyW = true; };

  root.GremlinGPU = { available: available, create: create, GpuBackend: GpuBackend };
})(typeof globalThis !== 'undefined' ? globalThis : this);
