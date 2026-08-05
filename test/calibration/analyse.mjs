/*
 * Read the sweeps written by sweep.mjs.
 *
 * Argmax alone is the wrong statistic: on a flat surface it is decided by one
 * contact, so a slope fitted through argmaxes looks confident when it is not.
 * This uses a within-subject design instead -- proteins differ enormously in
 * baseline (67% to 89% top-L/2), so each protein's own mean across multipliers
 * is subtracted and the DEVIATIONS are averaged. That answers "relative to this
 * protein's average, which multiplier does better", which is the question the
 * sweep can support.
 *
 * It also reports whether each curve has an INTERIOR maximum. A peak sitting on
 * a grid edge means the optimum was never bracketed, and any exponent fitted
 * through it is a bound rather than an estimate.
 *
 *   node analyse.mjs [--check=400]
 */
import { readFileSync, existsSync } from 'node:fs';

const HERE = new URL('.', import.meta.url).pathname;
function opt(n, d) {
  const h = process.argv.find(a => a.startsWith('--' + n + '='));
  return h === undefined ? d : h.slice(n.length + 3);
}
const CK = parseInt(opt('check', '400'), 10);

function load(arm) {
  const f = HERE + 'results/' + arm + '.jsonl';
  if (!existsSync(f)) return [];
  const by = new Map();
  for (const ln of readFileSync(f, 'utf8').split('\n')) {
    if (!ln.trim()) continue;
    let r; try { r = JSON.parse(ln); } catch (e) { continue; }
    by.set(r.key, r);          // dedup; repeats are bit-identical (fixed seed)
  }
  return [...by.values()];
}

/* mean and standard error of the within-protein deviation, per multiplier */
function curve(rows, mults, multKey) {
  const byAcc = new Map();
  for (const r of rows) {
    if (!byAcc.has(r.acc)) byAcc.set(r.acc, {});
    const v = r.at[CK] && r.at[CK].l2;
    if (v !== null && v !== undefined) byAcc.get(r.acc)[r[multKey]] = v;
  }
  const dev = mults.map(() => []);
  let n = 0;
  for (const [, cells] of byAcc) {
    if (mults.some(m => cells[m] === undefined)) continue;   // complete rows only
    const mean = mults.reduce((s, m) => s + cells[m], 0) / mults.length;
    mults.forEach((m, i) => dev[i].push(cells[m] - mean));
    n++;
  }
  if (!n) return null;
  const mean = dev.map(a => a.reduce((s, v) => s + v, 0) / a.length);
  const se = dev.map((a, i) => {
    const varr = a.reduce((s, v) => s + (v - mean[i]) ** 2, 0) / Math.max(1, a.length - 1);
    return Math.sqrt(varr / a.length);
  });
  let bi = 0;
  mean.forEach((v, i) => { if (v > mean[bi]) bi = i; });
  return { mean, se, n, bi, interior: bi > 0 && bi < mults.length - 1 };
}

function slope(xs, ys) {
  const n = xs.length;
  if (n < 3) return null;
  const mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
  return den > 0 ? num / den : null;
}

const fmt = (v, s) => ((v >= 0 ? '+' : '') + (v * 100).toFixed(1)
  + (s === undefined ? '' : '±' + (s * 100).toFixed(1)));

/* ---------------- lr arm: length axis ---------------- */
const lr = load('lr');
if (lr.length) {
  const M = [...new Set(lr.map(r => r.lrMult))].sort((a, b) => a - b);
  console.log('\n' + '='.repeat(76));
  console.log('lr = 0.1*log(B)/L   multiplier sweep across natural lengths  @' + CK + ' steps');
  console.log('=' .repeat(76));
  console.log('deviation from each protein\'s own mean, percentage points (+/- s.e.)\n');
  console.log('group'.padEnd(18) + 'n' + M.map(m => ('x' + m).padStart(11)).join(''));
  console.log('-'.repeat(76));
  const bins = [
    { name: 'all', f: () => true },
    { name: 'L < 120', f: r => r.L < 120 },
    { name: 'L 120-190', f: r => r.L >= 120 && r.L < 190 },
    { name: 'L >= 190', f: r => r.L >= 190 }
  ];
  for (const b of bins) {
    const c = curve(lr.filter(b.f), M, 'lrMult');
    if (!c) continue;
    console.log(b.name.padEnd(18) + String(c.n).padStart(1)
      + c.mean.map((m, i) => fmt(m, c.se[i]).padStart(11)).join(''));
  }
  const all = curve(lr, M, 'lrMult');
  if (all) {
    const i1 = M.indexOf(1);
    const gap = all.mean[all.bi] - all.mean[i1];
    const pooled = Math.sqrt(all.se[all.bi] ** 2 + all.se[i1] ** 2);
    console.log('\n  peak x' + M[all.bi] + ', vs the formula (x1): ' + fmt(gap)
      + 'pt (' + (pooled > 0 ? (gap / pooled).toFixed(1) : '?') + ' s.e.)');
    console.log('  ' + (pooled > 0 && Math.abs(gap / pooled) < 2
      ? 'not distinguishable from the formula -- a broad basin, nothing to gain'
      : 'distinguishable from the formula'));
  }
}

/* ---------------- alpha arm: depth axis ---------------- */
const al = load('alpha');
if (al.length) {
  const M = [...new Set(al.map(r => r.aMult))].sort((a, b) => a - b);
  const depths = [...new Set(al.map(r => r.rows))].sort((a, b) => (a || 1e9) - (b || 1e9));
  console.log('\n' + '='.repeat(94));
  console.log('lam_w = 0.5*alpha*(L-1)(A-1)/Meff   alpha multiplier by depth  @' + CK + ' steps');
  console.log('='.repeat(94));
  console.log('depth'.padEnd(7) + 'Meff'.padStart(7) + '  n'
    + M.map(m => ('x' + m).padStart(8)).join('') + '    peak  bracketed?');
  console.log('-'.repeat(94));
  const pts = [];
  for (const d of depths) {
    const sub = al.filter(r => r.rows === d);
    // only multipliers measured for every protein at this depth
    const accs = [...new Set(sub.map(r => r.acc))];
    const have = M.filter(m => accs.every(a => sub.some(r => r.acc === a && r.aMult === m)));
    const c = curve(sub, have, 'aMult');
    if (!c) continue;
    const meffs = accs.map(a => (sub.find(r => r.acc === a) || {}).Meff).filter(Boolean).sort((x, y) => x - y);
    const med = meffs[meffs.length >> 1];
    const line = M.map(m => {
      const k = have.indexOf(m);
      return k < 0 ? '' : fmt(c.mean[k]);
    });
    pts.push({ meff: med, m: have[c.bi], interior: c.interior });
    console.log(String(d || 'full').padEnd(7) + med.toFixed(0).padStart(7) + '  ' + c.n
      + line.map(s => s.padStart(8)).join('')
      + ('x' + have[c.bi]).padStart(8)
      + (c.interior ? '  yes' : '  NO (grid edge -- optimum not bracketed)'));
  }
  if (pts.length >= 3) {
    const s = slope(pts.map(p => Math.log(p.meff)), pts.map(p => Math.log(p.m)));
    console.log('\n  best multiplier ~ Meff^' + s.toFixed(2)
      + '   =>   lam_w ~ Meff^' + (s - 1).toFixed(2) + '   (reference uses Meff^-1.00)');
    if (pts.some(p => !p.interior)) {
      console.log('  NOTE: at least one peak sits on a grid edge, so this is a bound.');
    }
    // multiplier-free cross-check: how far does the absolute penalty actually move?
    const L0 = 97, A0 = 21;
    const lam = (p) => 0.5 * 0.01 * p.m * (L0 - 1) * (A0 - 1) / p.meff;
    const a = pts[0], b = pts[pts.length - 1];
    console.log('  cross-check on absolute lam_w (L=' + L0 + '): '
      + (lam(a) / lam(b)).toFixed(0) + 'x swing over ' + (b.meff / a.meff).toFixed(0)
      + 'x depth  =>  Meff^' + (Math.log(lam(b) / lam(a)) / Math.log(b.meff / a.meff)).toFixed(2));
  }
}

if (!lr.length && !al.length) console.log('no results yet -- run sweep.mjs (see README.md)');
