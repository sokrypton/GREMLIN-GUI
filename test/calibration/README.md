# Calibration sweep

Does the reference's calibration hold across protein length and MSA depth?

The precision numbers elsewhere in this repo come from one protein, which cannot
check a formula whose whole point is that it varies with L and Meff. This sweeps
a multiplier on each of the two calibrated formulas over many proteins:

```
lr    = 0.1·log(B)/L                 shrinks with length
λ_w   = 0.5·α·(L−1)(A−1)/Meff        grows with length, falls with depth
```

A calibrated formula has its best multiplier at ~1 everywhere. A drifting
optimum means the exponent is off.

## Running it

```sh
./fetch-data.sh                                   # ~40MB from AlphaFold DB, not committed

node sweep.mjs --arm=lr                           # 12 proteins x 5 multipliers, ~60 min
node sweep.mjs --arm=alpha                        # 6 proteins x 4 depths x 6 multipliers
node sweep.mjs --arm=alphaL --meff=400,1500 \
               --mults=0.125,0.25,0.5,1,2         # length scaling, depth held fixed

node analyse.mjs                                  # read results/*.jsonl
```

Slices run in parallel on separate cores, and every run is resumable — results
append to `results/<arm>.jsonl` and cells already present are skipped:

```sh
node sweep.mjs --arm=alpha --slice=0 --nslice=3 &
node sweep.mjs --arm=alpha --slice=1 --nslice=3 &
node sweep.mjs --arm=alpha --slice=2 --nslice=3 &
```

Fits are seed-deterministic, so a cell computed twice is bit-identical and
`analyse.mjs` dedups by key. `results/` here holds the 252 cells the README's
conclusions were drawn from.

## Two things worth knowing before changing the design

**Do not vary length by cropping.** It is the obvious approach and it fails. A
crop is not a shorter protein, it is an amputated fragment: most of each
residue's contact partners fall outside the window, so the conditional for
column *i* loses most of its true predictors *and* the structure has almost
nothing left to score against. A 40-column centred crop of P0A9B2 left 13 true
contacts among 630 pairs — top-L/2 is 4 hits out of 20 — against 372 among
11,935 for a whole protein at L=155. A 16× change in lr was indistinguishable at
that resolution. Length therefore varies by using naturally different proteins.
Depth still varies by subsampling, which genuinely does give a shallower
alignment of the same protein.

**Hold depth fixed when testing the length scaling.** `lam_w` carries both
`(L-1)(A-1)` and `1/Meff`, so comparing proteins of different length at whatever
depth they happen to have measures the two together — and deeper alignments
masquerade as longer ones. `--meff=<targets>` subsamples each protein to a target
Meff first. The row count is solved in closed form from the full-depth weights
(subsampling by `f` gives `cnt_n -> 1 + f*(cnt_n - 1)`), so it costs nothing, and
the Meff recorded is the real one measured afterwards.

**Do not read the argmax on its own.** On a flat surface the argmax is decided
by a single contact, and an exponent fitted through a row of such argmaxes looks
far more confident than the data supports. `analyse.mjs` therefore reports a
within-protein deviation curve with standard errors — proteins differ from 67%
to 89% top-L/2, so their baselines have to come out first — and flags whether
each peak is interior. A peak on a grid edge means the optimum was never
bracketed and any exponent through it is a bound, not an estimate.

## Data

Twelve E. coli proteins, 70–270 residues, each with an AlphaFold DB model and
the a3m that model was built from, so alignment and ground truth agree by
construction. All are deeper than 8k sequences so depth does not confound the
length axis — which is why P0ABE7 (128 residues, 647 sequences) is deliberately
absent.
