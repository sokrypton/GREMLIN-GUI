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
node test/backends.test.mjs    # WASM + the self-test gate          (21 tests)
node test/wgsl.test.mjs        # WGSL validation (needs naga; skips if absent)
node test/eval-precision.mjs   # contact precision vs an AlphaFold structure
```

`eval-precision.mjs` is how the four changes in "Matching the reference"
below were decided; it needs two AFDB files it does not ship and prints how to
fetch them.

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

### Sequence reweighting (Meff)

`w_n = 1/|{m : identity(n,m) ≥ 0.8}|`, `Meff = Σ w_n`. Count how many sequences
are near-duplicates of each sequence, and give each one that fraction of a vote.
It is O(N²L), and on a real alignment it is the entire startup cost.

The tempting move is to approximate it. The better one is to make it fast: the
comparison is byte equality, so `i8x16.eq` + `bitmask` + `popcount` does sixteen
positions in three instructions with no per-pair JS call. On the demo alignment
(15,688 × 155, threshold 0.8):

| | Meff | time |
| --- | --- | --- |
| exact, WASM SIMD | 5,910 | **1.6 s** |
| exact, scalar JS | 5,910 | 34.1 s |
| greedy clustering (JS) | 6,834 | 8.4 s |
| random reference subset, R=3000 | 8,029 | 11.4 s |

The SIMD pass is *exact* — bit-identical weights to the scalar loop, verified
across `L mod 16 ∈ {15, 0, 1, 15, 1, 12, 11}` where the padding is discounted —
and still 5× faster than the approximation it replaces. So the default does not
approximate at all when WASM is available, and whole-page load for `P0A7Y4` went
from ~13 s to **4.2 s**.

Clustering survives only as the no-WASM fallback, where the choice is 8.4 s
against 34.1 s. It is worth being clear that it is *a different definition*, not
a cheaper way to evaluate the same one: it forces a hard partition where the true
neighbourhoods overlap, so it can only over-count, and it reads 16% high here
(93% high on a synthetic set built to make neighbourhoods chain). The random
reference subset is gone — it was slower, more biased, and seed-dependent.

Contact precision is the same under all of them (80.6 / 79.2 / 68.4 at top L/5,
L/2, L), which is the honest summary: Meff is not sensitive enough here for the
approximation to matter. It just should not have been slower and biased for no
reason.

### Redundancy filter (max pairwise identity)

Optional greedy clustering: walk the sequences in order, keep one as a
representative, drop any later sequence more identical than the threshold to a
representative already kept. The query is always kept, and because rows are
sorted by identity to the query, the survivor of a cluster is the one most like
it. It runs in the worker with progress reporting, since it is O(N × reps) and
takes tens of seconds on a large alignment.

**It is off by default, and probably should stay off.** Meff reweighting already
down-weights near-duplicates, so this trades data for a smaller N rather than for
accuracy. On the demo alignment (400 steps, B=128):

| max identity | N | Meff | prep (parse+filter+reweight) | top L/5 | top L/2 | top L |
| --- | --- | --- | --- | --- | --- | --- |
| off | 15,688 | 5,910 | 2.3s | 80.6% | 79.2% | 68.4% |
| 0.99 | 15,435 | 5,913 | 4.9s | 80.6% | 79.2% | 67.7% |
| 0.95 | 14,604 | 5,934 | 11.4s | 80.6% | 77.9% | 69.0% |
| 0.90 | 12,528 | 5,993 | 15.1s | 80.6% | 79.2% | 68.4% |
| 0.80 | 6,834 | 6,834 | 10.8s | 80.6% | 79.2% | 68.4% |

Throwing away 20% of the alignment at 0.90 moves Meff from 5,910 to 5,993 — the
reweighting had already discounted exactly those sequences to near nothing.
Every row is within noise of the unfiltered baseline. The 0.80 row is the tell:
Meff comes out exactly equal to N, because filtering at threshold *T* makes
reweighting at *T* a no-op — the two mechanisms are doing the same job. Use the
filter when you want a smaller N (memory, parse time, exporting a non-redundant
set), not when you want better contacts.

That equality is not a coincidence, and the code now relies on it. The filter and
cluster-mode reweighting compute the *same* greedy partition and differ only in
what they do with it — the filter keeps one member per cluster and drops the
rest, reweighting keeps everyone at `1/|cluster|` — so both give
`Meff = #clusters`. And after filtering at *T*, every surviving pair is provably
below *T*, so a reweighting pass at any threshold ≥ *T* would return `Meff = N`
without finding anything. It is skipped outright rather than run: exact, not an
approximation. The skip deliberately does **not** fire below the filter
threshold, where pairs in [*T_reweight*, *T_filter*) still count — so the default
pairing of filter 0.9 with reweighting 0.8 still does the full work (Meff 5,993
at N 12,528, against 5,910 unfiltered).

#### Composition prefiltering: tried, measured, removed

Since every sequence occupies the same columns, the match count is bounded above
by `Σ_a min(count₁[a], count₂[a])` — each residue type can match only as often as
the rarer sequence contains it. The bound is exact and screens extremely well:
**95.9% of pairs pruned at id 0.90**. It still made things *slower*:

| threshold | exact only | + composition bound | pairs pruned |
| --- | --- | --- | --- |
| 0.90 | 9.74s | 9.92s (0.98×) | 95.9% |
| 0.80 | 10.51s | 12.53s (0.84×) | 27.2% |
| 0.70 | 3.47s | 4.93s (0.70×) | 2.6% |

The screen costs more than the test it screens. The byte-packed comparison bails
after `L − need + 1` mismatches — 16 of them at id 0.90, about five 32-bit word
comparisons — while the bound needs up to 21 integer mins. A one-operation
variant, `matches ≤ L − |gaps₁ − gaps₂|` (also exact), is cheap enough but prunes
only 6%, and was likewise a wash.

The same experiment against the Meff pass came out the same way: at threshold
0.8 the bound prunes 21.8% and costs 0.70×. Vectorizing the comparison, rather
than avoiding it, is what actually paid.

If this ever needs to be faster, the answer is an inverted k-mer index over the
representatives, CD-HIT style, rather than a tighter per-pair bound.

## What the objective is

```
J/Meff = (1/Meff) Σ_n w_n Σ_i −log p_i(x_ni)
         + (λ_w/Meff) Σ w²  +  (λ_b/Meff) Σ b²
```

with `w_n = 1/|{m : identity(n,m) ≥ 0.8}|` and `Meff = Σ w_n`.

Two regularization conventions, because the two pages want different things:

- **practical** (`regMode: 'gremlin'`) — `λ_w = 0.5·α(L−1)(A−1)`, `λ_b = β`. The
  `(L−1)(A−1)` factor is what lets one α setting behave the same at L=8 and
  L=256; a bare constant is only ever tuned for one size.
- **educational** (`regMode: 'raw'`) — `λ_w = α/2`, no L or Meff scaling, and no
  sequence reweighting (`Meff = N`). This reproduces the original's numbers, so
  the α/β/lr sliders still mean what they used to.

Minibatching samples sequences with probability `w_n/Meff`, which makes the cost
per step independent of N. N = 100k costs the same as N = 256. It also makes the
objective batch-invariant for free: the reference has to scale `lam` by `B/N`
when it minibatches because its data term is a sum, whereas ours is already a
per-sequence mean.

## Matching the reference

Checked against [`sokrypton/laxy`
`examples/gremlin_jax.ipynb`](https://github.com/sokrypton/laxy/blob/main/examples/gremlin_jax.ipynb)
and GREMLIN_TF v2.1.

The reference writes its objective as a *sum* with a constant penalty:

```python
lam = 0.01                                  # constant, no explicit normalizer
if batch_size is not None:
    lam *= batch_size/N
    learning_rate = 0.1*log(batch_size)/L
else:
    learning_rate = 0.1*log(N)/L
cce_loss = -(x*log(softmax(logits))).sum([1,2])
cce_loss *= w                               # w_n = 1/|neighbours at 80%|
l2_loss  = 0.5*(L-1)*(A-1)*sum(w**2) + sum(b**2)
loss     = cce_loss.sum() + lam*l2_loss     # SUM over sequences
```

`N` there is the effective sequence count — the data term is weighted, so its
scale is `Meff`, and that is the quantity the constant `lam` is implicitly
divided against. Writing the same objective as a per-sequence mean makes it
explicit: `λ_w = 0.5·α·(L−1)(A−1)/Meff` and `λ_b = β/Meff`.

`test/core.test.mjs` pins this rather than asserting it — it scores the
reference's loss directly from the same parameters and compares. On a
deliberately redundant alignment (`Meff = 32`, `N = 264`, so the two normalizers
are 8× apart) ours matches reference/Meff to **5e-9** and misses reference/N by
8×, which is what makes the claim testable at all. The reference's
`lam *= batch_size/N` is the same bookkeeping in reverse: its data term is a sum,
so it shrinks with the batch and the penalty must shrink with it, whereas our
mean is batch-invariant to begin with.

Its objective is therefore the *sum* over sequences and ours is the
Meff-normalized mean, so ours is exactly theirs divided by Meff — same minimizer,
and Adam is invariant to the constant.

Four things were wrong here and are now fixed. To decide rather than guess, each
was scored against the AlphaFold model for the demo protein (`P0A7Y4`, RNase H,
155 residues, 372 true contacts at CB < 8Å and |i−j| ≥ 5), 400 steps at B=128.
The second row has all four changes applied; each `ablate:` row below turns
exactly one of them back off:

| | top L/5 | top L/2 | top L |
| --- | --- | --- | --- |
| before | 77.4% | 75.3% | 64.5% |
| **all four changes** | **80.6%** | **79.2%** | **68.4%** |
| ablate: gaps back in the norm | 80.6% | 76.6% | 65.8% |
| ablate: bias starts at zero | 80.6% | 77.9% | 65.8% |
| ablate: old fixed lr 0.05 | 77.4% | 77.9% | 64.5% |
| ablate: 2× coupling penalty | 80.6% | 77.9% | 67.7% |

- **Gap state excluded from the contact norm.** The reference takes the
  Frobenius norm over the 20×20 amino-acid block — *"note: we ignore gaps"*.
  Gap couplings carry alignment and phylogeny signal, not structural contact.
- **Bias initialized from single-site log frequencies**, `log(counts +
  0.01·log N)` centred per column, instead of zeros. The conditionals then start
  out already explaining column composition.
- **The coupling penalty carries a 0.5.** Ours was 2× the reference at the same
  α.
- **Learning rate `0.1·log(batch)/L`.** The pseudo-likelihood sums L conditionals
  per sequence, so a rate tuned at L=48 overshoots at L=155 — the ablation row
  above is a fixed 0.05, which is 16× the formula at L=155. The practical page
  adopts the formula automatically until you move the slider. But see
  [below](#does-the-calibration-hold-across-length-and-depth): a 12-protein sweep
  finds the optimum is a *broad basin*, so this row is about not being far wrong
  rather than about the `1/L` exponent being finely tuned.

One deliberate deviation: we zero-sum gauge-fix each block before taking the
norm; the reference relies on L2 to pin the gauge implicitly. Re-measured under
exact Meff reweighting, gauge fixing is **indistinguishable** from the raw norm
on this benchmark — the top-L column differs by one pair out of 155. It stays
because it makes the norm gauge-independent by construction rather than by
trusting L2 to have pinned it, but the earlier claim here that it was "worth a
little" was reading noise:

| scoring | top L/5 | top L/2 | top L |
| --- | --- | --- | --- |
| reference: raw 20×20 norm | 80.6% | 79.2% | 69.0% |
| **ours: 20×20 + zero-sum gauge** | **80.6%** | **79.2%** | **68.4%** |
| 21×21 + gauge (gaps in) | 80.6% | 76.6% | 65.8% |
| raw 21×21, no gauge | 80.6% | 76.6% | 66.5% |

What the table *does* separate cleanly is excluding gaps from the norm, which is
worth ~3 points of top-L either way.

### What counts as a contact

Every precision number above is scored against **CB (CA for Gly) < 8 Å, |i−j| ≥
5** — the CASP convention, and the primary metric here so the numbers stay
comparable to everyone else's. But that definition is a single distance cutoff
with no structural context, and it is worth knowing how much it is doing.

The [solab contact page](https://github.com/sokrypton/solab)
(`assets/js/contact.js`) uses a considered alternative: a residue-residue
interaction needs the *side chains* pointing at each other, not just backbone
proximity, so each side chain is approximated by a point projected off the Cα
trace — along the bisector of the two Cα–Cα bonds, away from the backbone — at a
per-secondary-structure offset (H 3.0 Å, E 4.0 Å, L 3.5 Å), with a per-SS cutoff
(8.0 Å helix–helix, 8.5 Å otherwise). It was calibrated there against ConFind
contact degree > 0.01 over 151 native domains, scoring F1 0.78 versus ~0.3 for a
raw Cα cutoff. Secondary structure comes from TM-align's Cα-only `make_sec`,
ported in [`sokrypton/CIRPIN-web`](https://github.com/sokrypton/CIRPIN-web)
(`src/tmalign.js`) — Cα-only on both sides, so the two agree about what
information they may use. Both live in [`test/contacts.mjs`](test/contacts.mjs).

Same predictions, three ground truths:

| ground truth | true pairs | top L/5 | top L/2 | top L |
| --- | --- | --- | --- | --- |
| CB < 8.0 Å (CASP, primary) | 372 | 80.6% | 79.2% | 68.4% |
| CB < 8.6 Å (count-matched control) | 464 | 96.8% | 90.9% | 78.7% |
| solab per-SS virtual-Cβ | 465 | 90.3% | 93.5% | 80.0% |

The middle row is the control, and it is the point. The solab model calls 465
pairs contacts where CB < 8 Å calls 372, and a more permissive definition raises
precision for free — so the only fair comparison is against a plain cutoff tuned
to the same count. Once you do that, **most of the apparent jump is the
effective cutoff (8.0 → 8.6 Å), not the side-chain projection.** At matched
count the two trade places: solab is better at top L/2 and top L, worse at top
L/5, all within a couple of pairs on a single protein.

The honest reading is narrower than "GREMLIN is better than we thought", but not
nothing: of the top 155 predictions, 24 are pairs solab calls contacts and CB <
8 Å calls errors, and their CB–CB distances run 8.0–10.0 Å with a median of 8.4.
Those are just over the line. A 68% top-L against one definition and 80% against
another, on the same predictions, is mostly a statement about where the line was
drawn.

### Does the calibration hold across length and depth?

Everything above was scored on one protein. The reference's settings were tuned
across many, and its two length/depth-aware formulas are exactly the kind of
thing a single protein cannot check:

```
lr    = 0.1·log(B)/L                 shrinks with length
λ_w   = 0.5·α·(L−1)(A−1)/Meff        grows with length, falls with depth
```

So both were swept as a **multiplier** over 12 proteins (L = 70–270, AlphaFold DB
models plus the alignments those models were built from). A calibrated formula
has its best multiplier at ~1 everywhere; a drifting optimum means the exponent
is off. Each protein is its own control — baselines run 67% to 89% top-L/2, so
comparing raw numbers across proteins would drown the effect. Harness in
[`test/calibration/`](test/calibration/).

**Length varies by using different proteins, not by cropping.** Cropping is the
obvious design and it does not work: a crop is not a shorter protein, it is an
amputated fragment. Most of each residue's contact partners fall outside the
window, so the conditional for column *i* loses most of its true predictors and
the structure has almost nothing left to score against. A 40-column crop of
P0A9B2 left **13 true contacts among 630 pairs** — top-L/2 is then 4 hits out of
20 — against 372 among 11,935 for a whole protein at L=155. At that resolution a
16× change in lr is invisible.

#### lr: a broad basin, and the formula sits in it

Deviation from each protein's own mean, in points:

| group | n | ×0.25 | ×0.5 | ×1 | ×2 | ×4 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 12 | −0.9±1.3 | +0.0±0.6 | −0.2±0.4 | +0.9±0.9 | +0.1±0.8 |
| L < 120 | 6 | +0.1±1.9 | +0.7±0.3 | −0.4±0.7 | +0.3±1.2 | −0.7±1.0 |
| L 120–190 | 4 | +0.3±0.5 | +0.5±0.2 | −0.1±0.3 | −0.1±0.3 | −0.6±0.2 |
| L ≥ 190 | 2 | −6.0±5.0 | −3.3±3.3 | +0.3±1.3 | +4.7±3.6 | +4.3±3.3 |

A **16× range in lr moves top-L/2 by about a point**, and the peak is 1.1 s.e.
from the formula — not distinguishable. Nothing here beats `0.1·log(B)/L`, which
is the useful result: the formula is confirmed, and there is no tuning left to
do. It also bounds what the `1/L` term is worth. Over L = 70–270 the formula
itself only spans 3.8×, well inside a ±4× basin, so a single constant rate would
also have sat in the basin at every length tested. The `1/L` exponent earns its
place over a wider length range than this sweep covers, not within it.

#### α and length: `(L−1)(A−1)` is right

`λ_w` carries two scalings, so sweeping α across proteins of different length
only measures the length one if depth is held fixed — otherwise a deeper
alignment reads as a longer one. Depth is pinned by subsampling each protein to
a target Meff. Subsampling by *f* thins every neighbourhood, so
`cnt_n → 1 + f(cnt_n − 1)` and `Meff(f) ≈ f·Σ 1/(1 + f(cnt_n − 1))`; since
`cnt_n = 1/sw_n` from one full-depth pass, the row count that hits a target is
closed-form. It is not a small correction — reaching Meff ≈ 450 takes 744 rows at
L=70 and 856 at L=176.

Eight proteins, L = 70–196, at two matched depths:

| target Meff | short (L<120) peak | long (L≥120) peak | |
| --- | --- | --- | --- |
| 400 | ×1 | ×1 | identical |
| 1500 | ×0.125 | ×0.25 | adjacent, differ by 0.6pt at 1.0 s.e. |

**No length dependence survives**, so the `(L−1)(A−1)` factor already carries it
and there is nothing for a multiplier to fix. Per-protein argmaxes do wander
(one protein peaks at ×0.125 for one target and ×2 for the other) but those
curves are flat and the argmax is decided by a contact or two — the bin curves
are the statistic that means something here.

This also settles an earlier confusion. At full depth the short proteins looked
like they wanted ×0.125 and the long ones ×0.5, which would have implied the
length factor was under-correcting. That was depth leaking through: those long
proteins sat at lower Meff (P0A7B8 at 2025 against up to 8465 for the short
ones). With depth matched, the split disappears.

And it de-confounds the depth result below. The optimum still moves with depth at
fixed length — ×1 at Meff 400 down to ~×0.2 at Meff 1500 — so that effect is
depth, not length wearing a disguise.

#### α and depth: the reference's `1/Meff` under-corrects, but we keep it

Depth is varied by subsampling, which genuinely does give a shallower alignment
of the same protein. Six proteins, deviation from each protein's own mean:

| depth | Meff | ×0.0625 | ×0.125 | ×0.25 | ×0.5 | ×1 | ×2 | ×4 | ×8 | ×16 | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 seqs | 111 | −4.9 | −1.6 | −0.6 | +0.6 | +1.1 | +2.6 | **+2.9** | +1.2 | −1.4 | ×4 |
| 512 | 382 | −6.0 | −2.2 | +2.1 | +3.1 | **+5.0** | +4.9 | +2.1 | −2.6 | −6.4 | ×1 |
| 2048 | 1158 | −3.4 | +0.7 | +0.3 | +0.5 | **+2.1** | −0.2 | | | | ×1 |
| full | 4837 | −1.4 | **+2.2** | +1.3 | +1.2 | −1.0 | −2.4 | | | | ×0.125 |

Every curve turns over inside the grid, so each optimum is bracketed rather than
resting on an edge. The best multiplier scales as **Meff^−0.85**, which makes the
optimal penalty **λ_w ∝ Meff^−1.85** against the reference's Meff^−1. A
multiplier-free cross-check agrees: the absolute λ_w at the optimum swings 1391×
across a 43× depth range, i.e. Meff^−1.92.

**The shipped default stays at the reference — α = 0.01 with Meff^−1.** Three
reasons the measurement is not enough to move it:

- Depth here is a *subsample of a deep family*, not a genuinely shallow family. A
  random 128-sequence slice of a 15k-member family is still drawn from something
  diverse and well-populated; a family that only ever had 128 members is a
  different object, and that is the case a user with a shallow MSA is actually in.
- Six proteins, all L ≤ 129, at a fixed 400 steps.
- The gain is real but small — about 2–3 points at each depth extreme and nothing
  in the middle, where the curves are wide plateaus (at Meff = 382, ×0.5 through
  ×2 all sit within 2 points of the peak).

If it ever is adopted, the form should be pinned where α = 0.01 is already right
rather than silently rescaling everything — `λ_w = 0.5·α·(L−1)(A−1)/Meff ·
(Meff_ref/Meff)^0.85` with `Meff_ref ≈ 700`, which lands at or within ~1.7 points
of the peak at all four depths and leaves mid-depth alignments untouched.

### The GREMLIN_TF optimizer, tested and not adopted

GREMLIN_TF v2.1 uses a modified Adam that replaces the per-element second moment
with a single scalar per tensor — the running mean of the squared gradient
*norm* — and disables bias correction, so every element shares one normalizer
and the update keeps the gradient's direction. It is available here as
`optMode: 'gremlin'`, but it is not the default, because on this benchmark plain
per-element Adam is better at every step count and every learning rate tried:

| optimizer | lr | top L/5 | top L/2 | top L |
| --- | --- | --- | --- | --- |
| **Adam (per-element)** | 0.0031 | **80.6%** | **79.2%** | **68.4%** |
| GREMLIN_TF (scalar vt) | 0.5 | 80.6% | 72.7% | 65.8% |
| GREMLIN_TF (scalar vt) | 1.0 | 80.6% | 74.0% | 66.5% |
| GREMLIN_TF (scalar vt) | 2.0 | 80.6% | 75.3% | 66.5% |

The obvious explanation — that a scalar normalizer suffers from minibatch noise
in ‖g‖² — turned out to be wrong: repeating at B=1024 kept the same ordering
(Adam 71.0% top-L against 66.5%). It was designed as a full-batch L-BFGS
replacement, and that is where its advantage presumably lies.

Larger batches do help slightly per step (71.0% at B=1024/150 steps versus 69.0%
at B=128/400 steps) but not per second: at equal wall clock, B=128 for 400 steps
beats B=1024 for 50 steps, 69.0% against 63.9%.

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

Real example: `P0A7Y4` (L=155, 15,688 sequences after filtering, Meff = 5,910,
10.6M parameters, 162 MB) is ready 4.2 s after the file lands, runs ~2 steps/s
and settles in roughly two minutes.

**Rendering had to stop being DOM.** Element counts the original emitted:

| L | A | network `<line>` | weight `<rect>` |
| --- | --- | --- | --- |
| 3 | 3 | 54 | 81 |
| 64 | 21 | 1,778,112 | 1,806,336 |
| 128 | 21 | 7,168,896 | 7,225,344 |

Everything matrix-shaped is now a canvas; the network diagram draws only the
top-K couplings; W itself never crosses to the main thread. The UI stays
responsive (~20–30 ms click latency) with the worker saturated.

One trap in that last change, since it cost the diagram half its lines: W is
symmetric, so `topCouplings` enumerates the `i<j` half only — but a coupling
feeds *two* conditionals, pushing x'(j,b) from x(i,a) and x'(i,a) from x(j,b).
The original looped over all ordered pairs and got both directions for free;
top-K has to mirror each quintuple explicitly. Ranking is unaffected, since the
two copies share a magnitude.

### Reading the contact map

The contact map gets a **sequential** ramp — blank white through blue to near
black, one hue, light to dark — while the coupling matrix W keeps the signed
red/white/blue one. That split is not decoration: after APC the contact score is
a *magnitude*, and the diverging ramp paints its sign. Half the off-diagonal
cells come out slightly negative (p50 = −0.04× the colour anchor on the demo
alignment) and those are correction artifacts, so they belong in the blank, not
in red.

Two numbers set the ramp, and both were measured rather than eyeballed:

- **Anchor at the L-th ranked score, not at max.** APC output is heavily tailed —
  the top pair sits 4.9× the L-th here — so scaling to the maximum leaves
  everything except a handful of cells white.
- **γ = 1.6, not 0.7.** Only 6.8% of off-diagonal pairs reach a quarter of the
  anchor, so the map should read as mostly empty with the contacts standing out
  of it. The old γ = 0.7 *lifted* the low end — a 0.25× noise cell painted at
  0.38 intensity — which flooded the background with speckle that competed with
  the real arcs. At γ = 1.6 the same cell paints at 0.11 and the
  secondary-structure arcs are the first thing you see.

The top-L markers are gone, and the ramp is why. Anchoring at the L-th ranked
score means the top L pairs *are* the darkest cells, so circling them drew a
second copy of what the colour already said — and against a mostly-white map the
rings became the loudest thing on it. They were softened first and then dropped;
the ranked table beside the map gives the exact list when you want it.

#### Columns the gap filter removed still take up space

`max column gaps` drops columns the model then never sees, which shrinks L. That
is fine for fitting and wrong for drawing: closing the holes puts two positions
either side of a dropped column *adjacent to the diagonal*, which is the one
reading of a contact map you must not get wrong.

So `buildDataset` returns a display frame as well as the model columns. A column
dropped because the query has a gap there is not a query position at all and
never enters the frame; a column dropped by the gap-fraction filter is a query
position — skipped because too few sequences had a residue there, a fact about
the alignment rather than the protein — so it keeps its slot and comes back as a
blank row and column. The `L × L` result is scattered into the frame through
`colSlot`, hover reports "not modelled" rather than a `0.000` that would read as
a measured non-contact, and the ticks, table and CSV stay in input-alignment
numbering throughout.

Sequence separation follows the same rule: `min |i-j|` is measured in input
columns, not model columns. Compression means two residues 6 apart in the protein
can be 4 apart in the model, and a model-space cut would silently discard a
genuinely long-range pair.

On the demo alignment at `max column gaps = 0.3`, L drops 155 → 136 (the removed
columns are the terminal tails, 0–3 and 140–154) and the map still draws 155 × 155
with 19 blank. The top-ranked pairs come out at the same input columns as with
the filter off — 65/118, 85/107, 56/105 — which is the check that the mapping is
actually carrying through. With the filter off, `frameL === L` and every mapping
is the identity, so the default view is untouched.

## Where the remaining headroom is

WASM SIMD collected the ~4× that was available from vectorizing a single thread.
What is left:

- **Multiple workers.** The step parallelizes over positions with no shared
  writes until the symmetrize pass, so 4 workers should be close to linear. This
  is the largest untapped win that can actually be tested in a browser today.
- **WebGPU**, once someone runs it on real hardware. The step is two
  matmul-shaped passes, ~29 GFLOP at L=128/N=1000, i.e. tens of milliseconds on
  an integrated GPU.
- **The redundancy filter** is still scalar JS — it is the one remaining place
  that spends double-digit seconds on a large alignment (+6s at 0.9). It is the
  same byte comparison `meff_counts` vectorizes, so the same treatment applies;
  it has not had it because the filter is off by default.
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

Every contact-precision number in this README comes from one script, which needs
two files it does not ship:

```sh
curl -o af.pdb   https://alphafold.ebi.ac.uk/files/AF-P0A7Y4-F1-model_v6.pdb
curl -o test.a3m https://alphafold.ebi.ac.uk/files/msa/AF-P0A7Y4-F1-msa_v6.a3m
node test/eval-precision.mjs .           # ablations, plus all three ground truths
```
