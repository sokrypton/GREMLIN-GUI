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
style.css                     all of it
test/                         node tests and benchmarks
```

`gremlin-core.js` and `msa.js` load in node as well as in a worker, so the
numerics are testable without a browser:

```sh
node test/core.test.mjs
```

The suite checks the gradient against the *original* implementation
(`test/naive.mjs`, kept verbatim as an oracle), the invariants, and end-to-end
contact recovery on a synthetic alignment with planted couplings.

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

Scalar JS is roughly at its limit. The step is two matmul-shaped passes, so
WASM SIMD plus a few workers is worth ~4–8×, and WebGPU turns the whole step
into two matmuls (~29 GFLOP at L=128/N=1000, i.e. tens of milliseconds on an
integrated GPU). Sequence reweighting is O(N²L) and is the other place a GPU
would pay off — it is currently ~10 s for 15k sequences and is approximated
above 3000 (a biased approximation: Meff reads 8,099 at 3000 references versus
11,406 at 500).

## Benchmarks

The two measurements the design rests on, as standalone scripts:

```sh
node test/bench-original-vs-onehot.mjs   # the O(A) waste, and element counts
node test/bench-loop-order.mjs           # sequence-major vs position-major
```

Both verify their fast path against the original before timing it, so the
speedups are like-for-like rather than a different computation.
