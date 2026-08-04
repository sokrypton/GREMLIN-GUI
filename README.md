# GREMLIN in the browser

Pseudo-likelihood Potts model (plmDCA) fitted in a Web Worker, with couplings
turned into a contact map. Two pages, one backend:

| page | what it's for |
| --- | --- |
| [`index.html`](index.html) | **Educational.** The original toy visualization — every intermediate shown for a 4×3 digit alignment: one-hot input, couplings, bias, per-position softmax, the coupling matrix and the contact map. |
| [`practical.html`](practical.html) | **Practical.** Real alignments from AlphaFold DB or your own FASTA/A3M, with one output: `APC(L2norm(gauge-fixed w))`, redrawn live as it converges. |

No build step, no dependencies, no CDN. Workers need a real origin, so serve it:

```sh
python3 -m http.server 8000
# then open http://localhost:8000/
```

## Files

```
index.html   practical.html   the two pages
edu.js       practical.js     their UI
ui.js                         shared canvas drawing + formatting
msa.js                        FASTA/A3M parsing, filtering, encoding
gremlin-core.js               the numeric core; runs as a Worker
gremlin.c  gremlin.wasm       WASM SIMD backend (.wasm is committed)
gremlin-gpu.js                WebGPU backend + WGSL shaders
style.css                     all of it
test/                         node tests and benchmarks
```

`gremlin-core.js` and `msa.js` load in node as well as in a worker, so the
numerics are testable without a browser:

```sh
node test/core.test.mjs        # numerics, MSA parsing, cost model  (26 tests)
node test/backends.test.mjs    # WASM + the self-test gate          (16 tests)
node test/wgsl.test.mjs        # WGSL validation (needs naga; skips if absent)
```

`core.test.mjs` checks the gradient against the *original* implementation
(`test/naive.mjs`, kept verbatim as an oracle), the invariants, and end-to-end
contact recovery on a synthetic alignment with planted couplings.

To rebuild the WASM module after editing `gremlin.c` (needs clang with a wasm32
target; `gremlin.wasm` is committed so users never have to):

```sh
./build-wasm.sh
```

## Getting an alignment

The practical page fetches the a3m that AlphaFold DB used for its own prediction:

```
https://alphafold.ebi.ac.uk/files/msa/AF-<ACCESSION>-F1-msa_v6.a3m
```

CORS is open, so no proxy is needed. Enter a UniProt accession (the **Demo**
button loads `P0A7Y4`, *E. coli* RNase H) or upload a file.

Filtering follows [py2Dmol](https://github.com/sokrypton/py2Dmol) so the two
agree on what an alignment is: a3m lowercase columns are insertions and are
stripped; coverage is the non-gap fraction of a row; identity is measured
against the query; `-`, `.`, ` ` and `X` all count as gaps. Defaults are
coverage ≥ 0.75 and identity ≥ 0.15, sorted by identity with the query first.

## What the objective is

```
J/Meff = (1/Meff) Σ_n w_n Σ_i −log p_i(x_ni)
         + (λ_w/Meff) Σ w²  +  (λ_b/Meff) Σ b²
```

with `w_n = 1/|{m : identity(n,m) ≥ 0.8}|` and `Meff = Σ w_n`.

Two regularization conventions, because the two pages want different things:

- **practical** (`regMode: 'gremlin'`) — `λ_w = α(L−1)(A−1)`, `λ_b = β`. The
  `(L−1)(A−1)` factor is what lets one α setting behave the same at L=8 and
  L=256; a bare constant is only ever tuned for one size.
- **educational** (`regMode: 'raw'`) — `λ_w = α/2`, no L or Meff scaling, and no
  sequence reweighting (`Meff = N`). This reproduces the original's numbers, so
  the α/β/lr sliders still mean what they used to.

Minibatching samples sequences with probability `w_n/Meff`, which makes the cost
per step independent of N. N = 100k costs the same as N = 256.

## Backends

The step runs on the fastest of three backends, picked at startup:

| backend | status | speed at L=155, B=128 |
| --- | --- | --- |
| **WebGPU** (`gremlin-gpu.js`) | shaders validated, execution **unverified** — see below | not measured |
| **WASM SIMD** (`gremlin.c` → `gremlin.wasm`) | measured | 117 ms/step (**3.9×** JS) |
| **plain JS** (`gremlin-core.js`) | the reference | 462 ms/step |

In the browser the WASM path takes the real `P0A7Y4` alignment from 2.1 to
**7.1 steps/s**, so it settles in well under a minute instead of two.

The active backend is shown as a badge next to the Start button; hover it to see
what was tried and why.

### Every backend has to prove itself first

A candidate is not used until it reproduces the plain-JS reference — gradient,
bias gradient, loss, updated parameters and contact map — on a small fixed probe
problem, within 2e-3. A backend that disagrees refuses itself and the next one
down is tried. JS is always available and is the reference, so selection cannot
fail.

This is not ceremony. A miscompiled kernel or a wrong workgroup index produces a
*plausible-looking* contact map that is quietly wrong, which is the worst failure
mode this tool has. `test/backends.test.mjs` checks the gate actually fires, by
sabotaging one kernel at a time:

| sabotage | deviation | verdict |
| --- | --- | --- |
| coupling gradient scaled by 1.05 | 7.3e-2 | rejected |
| symmetrization skipped | 8.2 | rejected |
| Adam learning rate off by 1.5× | 2.5 | rejected |
| contact map offset by 0.05 | 5.0e+2 | rejected |
| *(the real WASM backend)* | *1.3e-5* | *accepted* |

The parameter check is deliberately 25× looser than the gradient check. Adam's
first step is `lr·g/(|g|+eps)`, essentially `lr·sign(g)`, which is discontinuous
at zero — so wherever the true gradient is near zero, a 7e-6 difference between
two *correct* implementations can move that parameter by a full learning rate.
Holding W as tightly as G would reject correct backends.

### About the WebGPU backend

It has never been executed. The environment this was written in has no WebGPU:
Playwright's Chromium ships with the API disabled (`navigator.gpu` is undefined
under every flag combination, and there is no Vulkan driver to fall back on).

What it does have: all nine WGSL shaders are validated by
[naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) in `test/wgsl.test.mjs`,
which catches syntax, type, binding and control-flow errors and lowers each to
SPIR-V. That does **not** catch a wrong index expression — which is exactly what
the self-test gate above is for. The failure mode if a shader is wrong is that
the probe disagrees, WebGPU refuses itself, and you silently get WASM.

Design notes, for whoever finishes it on real hardware:

- W, G, m and v stay in device buffers for the whole run; W is 162MB at L=155, so
  the point is that it never crosses the bus. Only the L×L contact map (96KB)
  and a few scalars are read back.
- The backward pass avoids f32 atomics (which WGSL does not have) by giving one
  workgroup one ordered position pair and keeping the whole A×A block in
  registers: thread `a` holds `acc[b]` for all b, loops the batch accumulating
  `acc[x_nj] += d[n][i][a]`, then writes A² values once.
- `stepAsync` only encodes and submits — it does not block. The periodic contact
  map readback in the snapshot path doubles as the queue drain, which is what
  stops submissions running arbitrarily far ahead of the device.

## Performance notes

Measured on this machine (4 cores, scalar JS, single worker).

**The math was O(A) wasteful.** The original built a dense one-hot vector and
reduced over it, making the forward pass O(L²A²) per sequence when the input has
only L non-zeros. Skipping the zeros — same result, verified to 1e-16 — gives:

| L | A | N | original | one-hot | speedup |
| --- | --- | --- | --- | --- | --- |
| 16 | 21 | 64 | 85.8 ms | 2.5 ms | 35× |
| 24 | 21 | 128 | 431.7 ms | 15.1 ms | 29× |
| 32 | 21 | 256 | 1243.9 ms | 48.6 ms | 26× |

**It was then cache-bound, not FLOP-bound.** Iterating sequence-major touches
L(L−1) scattered A×A blocks per sequence, streaming all of W once per sequence.
Position-major (positions outer, sequences inner) reads W linearly against a
small hot buffer — 2.3× more, with bit-identical gradients.

**Memory is the ceiling on L, not arithmetic.** Full-rank Potts is inherently
O(L²A²) parameters, and Adam needs four copies (W, gradient, m, v):

| L | params | W+G+m+v (f32) |
| --- | --- | --- |
| 64 | 1.8 M | 28 MB |
| 128 | 7.2 M | 110 MB |
| 256 | 28.9 M | 441 MB |
| 384 | 65.0 M | 992 MB |

L ≈ 250 is comfortable, L ≈ 400 is the practical limit, and the page refuses
past 1.6 GB rather than crashing the tab. Packing the `i<j` half would halve it.

**That ceiling also shows up as time.** The Adam update is O(L²A²) per step
*regardless of batch size* — a ~100 ms floor at L=155. So throughput saturates:
252 seq/s at B=64, 330 at B=128, 361 at B=256, 410 at B=1024. The practical page
defaults to B=128, which buys 91% of peak throughput while giving the contact map
three times as many frames to evolve through.

Real example: `P0A7Y4` (L=155, 15,688 sequences after filtering, Meff≈8,100,
10.6M parameters, 162 MB) runs ~2 steps/s and settles in roughly two minutes.

**Rendering had to stop being DOM.** Element counts the original emitted:

| L | A | network `<line>` | weight `<rect>` |
| --- | --- | --- | --- |
| 3 | 3 | 54 | 81 |
| 64 | 21 | 1,778,112 | 1,806,336 |
| 128 | 21 | 7,168,896 | 7,225,344 |

Everything matrix-shaped is now a canvas; the network diagram draws only the
top-K couplings; W itself never crosses to the main thread. The UI stays
responsive (~20–30 ms click latency) with the worker saturated.

## Where the remaining headroom is

WASM SIMD collected the ~4× that was available from vectorizing a single thread.
What is left:

- **Multiple workers.** The step parallelizes over positions with no shared
  writes until the symmetrize pass, so 4 workers should be close to linear. This
  is the largest untapped win that can actually be tested in a browser today.
- **WebGPU**, once someone runs it on real hardware. The step is two
  matmul-shaped passes, ~29 GFLOP at L=128/N=1000, i.e. tens of milliseconds on
  an integrated GPU.
- **Sequence reweighting** is O(N²L) and still ~10 s for 15k sequences even after
  byte-packing the comparison. It is approximated above 3000 references, and that
  approximation is biased upward — Meff reads 8,099 at 3000 references versus
  11,406 at 500, because 1/count is convex. This is embarrassingly parallel and
  would suit either a worker pool or the GPU.
- **Packing the `i<j` half of W** would halve the memory ceiling, taking L=384
  from 992MB to about 520MB.

## Benchmarks

The two measurements the design rests on, as standalone scripts:

```sh
node test/bench-original-vs-onehot.mjs   # the O(A) waste, and element counts
node test/bench-loop-order.mjs           # sequence-major vs position-major
```

Both verify their fast path against the original before timing it, so the
speedups are like-for-like rather than a different computation.
