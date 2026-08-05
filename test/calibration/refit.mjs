/*
 * Fit the depth exponent from ALL the alpha data at once.
 *
 * This became possible only after the other two formulas were confirmed. lr is
 * flat, so it is not interacting with anything; (L-1)(A-1) is right, so length
 * is not a confounder and cells from proteins of different length can be pooled.
 * What is left is one free parameter.
 *
 * Model.  Suppose the best multiplier on alpha scales as
 *     m*(Meff) = C * (Meff/Meff_ref)^s
 * Then for the right s, rescaling every cell's multipliers by
 *     x = log2(m) - s*log2(Meff/Meff_ref)
 * should line every cell's curve up on a single peak. So: grid-search s, and for
 * each candidate fit ONE quadratic in x through all the pooled points with a
 * per-cell intercept (a cell being one protein at one depth). The s that
 * minimizes residual error is the estimate, and the fitted vertex gives C.
 *
 * This replaces reading a per-cell argmax and regressing through those. The
 * curves are flat enough that an argmax moves on one contact -- the length arm
 * produced argmax slopes of L^-0.21 and L^-0.96 from curves whose bins agreed
 * exactly. Fitting the shape uses every point instead of the noisiest one.
 *
 * Uncertainty is a bootstrap over PROTEINS, not over cells: cells from the same
 * protein share a structure, an alignment and a contact definition, so they are
 * not independent draws.
 *
 *   node refit.mjs [--check=400] [--ref=700]
 */
import { readFileSync, existsSync } from 'node:fs';

const HERE = new URL('.', import.meta.url).pathname;
function opt(n, d) {
  const h = process.argv.find(a => a.startsWith('--' + n + '='));
  return h === undefined ? d : h.slice(n.length + 3);
}
const CK = parseInt(opt('check', '400'), 10);
const MREF = parseFloat(opt('ref', '700'));

/* ---- pool every alpha measurement ---- */
const cells = new Map();          // cell key -> { acc, Meff, L, pts: [[mult, prec]] }
for (const arm of ['alpha', 'alphaL']) {
  const f = HERE + 'results/' + arm + '.jsonl';
  if (!existsSync(f)) continue;
  const seen = new Set();
  for (const ln of readFileSync(f, 'utf8').split('\n')) {
    if (!ln.trim()) continue;
    let r; try { r = JSON.parse(ln); } catch (e) { continue; }
    if (seen.has(r.key)) continue;
    seen.add(r.key);
    const v = r.at[CK] && r.at[CK].l2;
    if (v === null || v === undefined) continue;
    const k = arm + '|' + r.acc + '|' + r.rows + '|' + (r.target ?? '');
    if (!cells.has(k)) cells.set(k, { acc: r.acc, L: r.L, Meff: r.Meff, pts: [] });
    cells.get(k).pts.push([r.aMult, v]);
  }
}
const all = [...cells.values()].filter(c => c.pts.length >= 4);   // need a shape
const accs = [...new Set(all.map(c => c.acc))];
console.log('pooled ' + all.length + ' cells (protein x depth) over ' + accs.length
  + ' proteins, ' + all.reduce((s, c) => s + c.pts.length, 0) + ' fits, @' + CK + ' steps');
console.log('Meff spans ' + Math.min(...all.map(c => c.Meff)).toFixed(0)
  + ' to ' + Math.max(...all.map(c => c.Meff)).toFixed(0)
  + ', L spans ' + Math.min(...all.map(c => c.L)) + ' to ' + Math.max(...all.map(c => c.L)));

/*
 * For a candidate s: build x per point, demean x, x^2 and y WITHIN each cell
 * (that is the per-cell intercept), then least-squares y ~ a*x^2 + b*x.
 */
function fitAt(s, subset) {
  let sxx = 0, sxy = 0, sxz = 0, szz = 0, szy = 0, syy = 0, n = 0;
  for (const c of subset) {
    const xs = c.pts.map(p => Math.log2(p[0]) - s * Math.log2(c.Meff / MREF));
    const ys = c.pts.map(p => p[1]);
    const zs = xs.map(x => x * x);
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const mz = zs.reduce((a, b) => a + b, 0) / zs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    for (let i = 0; i < xs.length; i++) {
      const X = xs[i] - mx, Z = zs[i] - mz, Y = ys[i] - my;
      sxx += X * X; sxz += X * Z; szz += Z * Z;
      sxy += X * Y; szy += Z * Y; syy += Y * Y; n++;
    }
  }
  // solve [[szz,sxz],[sxz,sxx]] [a,b] = [szy,sxy]
  const det = szz * sxx - sxz * sxz;
  if (Math.abs(det) < 1e-18) return null;
  const a = (szy * sxx - sxy * sxz) / det;
  const b = (sxy * szz - szy * sxz) / det;
  const rss = syy - (a * szy + b * sxy);
  return { a, b, rss, n, vertex: a < 0 ? -b / (2 * a) : null };
}

function bestS(subset) {
  let best = null;
  for (let s = -2.5; s <= 1.0; s += 0.005) {
    const f = fitAt(s, subset);
    if (!f || f.a >= 0) continue;                 // must be concave to have a peak
    if (!best || f.rss < best.rss) best = Object.assign({ s }, f);
  }
  return best;
}

const fit = bestS(all);
if (!fit) { console.log('no concave fit found'); process.exit(0); }

/* bootstrap over proteins */
let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const boot = [];
for (let b = 0; b < 400; b++) {
  const pick = [];
  for (let i = 0; i < accs.length; i++) pick.push(accs[(rnd() * accs.length) | 0]);
  const sub = [];
  for (const a of pick) for (const c of all) if (c.acc === a) sub.push(c);
  const f = bestS(sub);
  if (f) boot.push(f.s);
}
boot.sort((a, b) => a - b);
const lo = boot[Math.floor(boot.length * 0.025)], hi = boot[Math.floor(boot.length * 0.975)];

const C = Math.pow(2, fit.vertex);
console.log('\n' + '='.repeat(72));
console.log('best multiplier  m*(Meff) = ' + C.toFixed(3) + ' * (Meff/' + MREF + ')^' + fit.s.toFixed(2));
console.log('  95% CI on the exponent (bootstrap over proteins): ['
  + lo.toFixed(2) + ', ' + hi.toFixed(2) + ']');
console.log('  => lam_w ~ Meff^' + (fit.s - 1).toFixed(2)
  + '   [' + (lo - 1).toFixed(2) + ', ' + (hi - 1).toFixed(2) + ']');
console.log('  reference is Meff^-1.00, i.e. exponent 0 on the multiplier'
  + (lo < 0 && hi < 0 ? '  -- EXCLUDED by the CI' : '  -- inside the CI'));
console.log('='.repeat(72));

/* what the correction is worth, per cell, from the fitted quadratic */
console.log('\npredicted gain over the reference (fitted curve, points of top-L/2)');
console.log('Meff'.padStart(7) + '  n' + '   m* '.padStart(8) + '   gain');
console.log('-'.repeat(40));
const byDepth = new Map();
for (const c of all) {
  const bucket = Math.round(Math.log2(c.Meff) * 2) / 2;
  if (!byDepth.has(bucket)) byDepth.set(bucket, []);
  byDepth.get(bucket).push(c);
}
let tot = 0, totN = 0;
for (const k of [...byDepth.keys()].sort((a, b) => a - b)) {
  const grp = byDepth.get(k);
  const meff = grp.reduce((s, c) => s + c.Meff, 0) / grp.length;
  const shift = fit.s * Math.log2(meff / MREF);
  const mStar = Math.pow(2, fit.vertex + shift);
  // deviation of the fitted parabola at m* versus at the reference m = 1
  const xStar = fit.vertex, xRef = 0 - shift;
  const y = (x) => fit.a * x * x + fit.b * x;
  const gain = y(xStar) - y(xRef);
  tot += gain * grp.length; totN += grp.length;
  console.log(meff.toFixed(0).padStart(7) + '  ' + String(grp.length).padStart(2)
    + ('x' + mStar.toFixed(3)).padStart(9) + (gain * 100).toFixed(1).padStart(7) + 'pt');
}
console.log('-'.repeat(40));
console.log('weighted mean gain'.padEnd(21) + (tot / totN * 100).toFixed(1).padStart(7) + 'pt');
console.log('\nproposed form, pinned so mid-depth alignments are untouched:');
console.log('  lam_w = 0.5*alpha*(L-1)(A-1)/Meff * (' + MREF + '/Meff)^' + (-fit.s).toFixed(2));
