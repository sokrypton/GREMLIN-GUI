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

/* ---------------- alphaL arm: length axis, depth held fixed ---------------- */
const aL = load('alphaL');
if (aL.length) {
  const M = [...new Set(aL.map(r => r.aMult))].sort((a, b) => a - b);
  const targets = [...new Set(aL.map(r => r.target))].sort((a, b) => a - b);
  console.log('\n' + '='.repeat(86));
  console.log('alpha multiplier vs LENGTH at matched depth  @' + CK + ' steps');
  console.log('='.repeat(86));
  console.log('the (L-1)(A-1) factor is only testable with Meff held fixed; otherwise a');
  console.log('deeper alignment reads as a longer one\n');
  for (const t of targets) {
    const sub = aL.filter(r => r.target === t);
    console.log('--- target Meff ' + t + ' ---');
    console.log('protein'.padEnd(10) + 'L'.padStart(5) + 'Meff'.padStart(7)
      + M.map(m => ('x' + m).padStart(8)).join('') + '    peak');
    console.log('-'.repeat(86));
    const pts = [];
    const accs = [...new Set(sub.map(r => r.acc))]
      .sort((a, b) => (sub.find(r => r.acc === a).L) - (sub.find(r => r.acc === b).L));
    for (const acc of accs) {
      const cells = M.map(m => {
        const r = sub.find(x => x.acc === acc && x.aMult === m);
        return r && r.at[CK] ? r.at[CK].l2 : null;
      });
      if (cells.some(c => c === null)) continue;
      let bi = 0;
      cells.forEach((c, i) => { if (c > cells[bi]) bi = i; });
      const row = sub.find(r => r.acc === acc);
      pts.push({ L: row.L, m: M[bi] });
      console.log(acc.padEnd(10) + String(row.L).padStart(5) + row.Meff.toFixed(0).padStart(7)
        + cells.map(c => (c * 100).toFixed(1).padStart(8)).join('')
        + ('x' + M[bi]).padStart(8));
    }
    // within-protein curve, and whether its peak moves with L
    const shortC = curve(sub.filter(r => r.L < 120), M, 'aMult');
    const longC = curve(sub.filter(r => r.L >= 120), M, 'aMult');
    for (const [nm, c] of [['L < 120', shortC], ['L >= 120', longC]]) {
      if (!c) continue;
      console.log('  ' + nm.padEnd(10) + ' n=' + c.n + '  deviation'
        + c.mean.map((v, i) => fmt(v, c.se[i]).padStart(12)).join('')
        + '   peak x' + M[c.bi]);
    }
    /*
     * The per-protein argmax slope is reported but must not be read as an
     * exponent: these curves are flat, so each protein's argmax is decided by a
     * contact or two and jumps around (a protein peaking at x0.125 for one
     * target and x2 for the other is jitter, not length dependence). The
     * question that the data can answer is whether the SHORT and LONG bin
     * curves peak in the same place -- if they do, (L-1)(A-1) already carries
     * the length dependence and there is nothing left for a multiplier to fix.
     */
    const s = slope(pts.map(p => Math.log(p.L)), pts.map(p => Math.log(p.m)));
    if (s !== null) {
      console.log('  per-protein argmax slope: L^' + s.toFixed(2)
        + '  (jitter on flat curves -- do not read as an exponent)');
    }
    if (shortC && longC) {
      const same = M[shortC.bi] === M[longC.bi];
      const gap = longC.mean[longC.bi] - longC.mean[shortC.bi];
      const pooled = Math.sqrt(longC.se[longC.bi] ** 2 + longC.se[shortC.bi] ** 2);
      console.log('  short peaks x' + M[shortC.bi] + ', long peaks x' + M[longC.bi]
        + (same ? '  -- SAME: no length dependence left over'
                : '  -- differ by ' + fmt(gap) + 'pt ('
                  + (pooled > 0 ? (gap / pooled).toFixed(1) : '?') + ' s.e.'
                  + (pooled > 0 && Math.abs(gap / pooled) < 2 ? ', not distinguishable' : '') + ')'));
    }
    console.log('');
  }
}

/* ---------------- holdout: the correction against plain 1/Meff ------------- */
const ho = load('holdout');
if (ho.length) {
  console.log('\n' + '='.repeat(84));
  console.log('HELD-OUT head-to-head: does the depth correction beat plain 1/Meff?  @400 steps');
  console.log('='.repeat(84));
  console.log('proteins fetched after the exponent was fitted; paired by protein x depth,');
  console.log('same subsample, same seed, only alpha differs\n');
  const cells = new Map();
  for (const r of ho) {
    const k = r.acc + '|' + r.rows;
    if (!cells.has(k)) cells.set(k, { acc: r.acc, L: r.L, rows: r.rows, Meff: r.Meff, s: {} });
    cells.get(k).s[r.setting] = r.p;
  }
  const complete = [...cells.values()].filter(c => c.s.ref && c.s.expo && c.s.full);
  const groups = [
    { name: 'shallow (subsampled)', f: (c) => c.rows > 0 },
    { name: 'full depth', f: (c) => c.rows === 0 },
    { name: 'all cells', f: () => true }
  ];
  const metrics = [['top L/5', 'l5'], ['top L/2', 'l2'], ['top L', 'l']];
  for (const g of groups) {
    const sub = complete.filter(g.f);
    if (!sub.length) continue;
    console.log('--- ' + g.name + ' (n=' + sub.length + ') ---');
    console.log('metric'.padEnd(10) + 'ref'.padStart(8) + 'expo'.padStart(8) + 'full'.padStart(8)
      + '     expo-ref        full-ref');
    for (const [nm, key] of metrics) {
      const mean = (s) => sub.reduce((a, c) => a + c.s[s][key], 0) / sub.length;
      const paired = (s) => {
        const d = sub.map(c => c.s[s][key] - c.s.ref[key]);
        const m = d.reduce((a, b) => a + b, 0) / d.length;
        const v = d.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, d.length - 1);
        return { m, se: Math.sqrt(v / d.length), win: d.filter(x => x > 1e-9).length,
                 loss: d.filter(x => x < -1e-9).length };
      };
      const e = paired('expo'), f = paired('full');
      const show = (p) => ((p.m >= 0 ? '+' : '') + (p.m * 100).toFixed(2) + '±'
        + (p.se * 100).toFixed(2) + ' (' + p.win + 'W/' + p.loss + 'L)').padStart(16);
      console.log(nm.padEnd(10)
        + (mean('ref') * 100).toFixed(1).padStart(8)
        + (mean('expo') * 100).toFixed(1).padStart(8)
        + (mean('full') * 100).toFixed(1).padStart(8)
        + show(e) + show(f));
    }
    console.log('');
  }
  console.log('a coefficient whose CI excludes zero and a formula that predicts better');
  console.log('are different claims; this table is the second one.');
}

if (!lr.length && !al.length && !aL.length && !ho.length) {
  console.log('no results yet -- run sweep.mjs (see README.md)');
}
