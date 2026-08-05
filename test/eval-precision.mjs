/*
 * Contact-precision evaluation against the AlphaFold structure for P0A7Y4
 * (E. coli RNase H, 155 residues -- same length as the AFDB a3m).
 *
 * Ground truth: CB-CB (CA for Gly) < 8 A, |i-j| >= 5.
 * Metric: precision in the top L and top L/2 predicted pairs, which is the
 * standard way DCA contact predictions are scored.
 *
 * Purpose: decide empirically whether aligning to the gremlin_jax reference
 * (gap-excluded norm, frequency bias init, 0.5 factor on the coupling penalty,
 * L-aware learning rate) actually predicts contacts better.
 */
import { readFileSync, existsSync } from 'node:fs';

/*
 * Needs two files this repo does not ship (they are ~4MB and not ours):
 *   af.pdb    https://alphafold.ebi.ac.uk/files/AF-P0A7Y4-F1-model_v6.pdb
 *   test.a3m  https://alphafold.ebi.ac.uk/files/msa/AF-P0A7Y4-F1-msa_v6.a3m
 * Put them next to this script, or pass a directory as argv[2].
 */
const DIR = process.argv[2] || '.';
if (!existsSync(DIR + '/af.pdb') || !existsSync(DIR + '/test.a3m')) {
  console.log('skipping: af.pdb and test.a3m not found in ' + DIR);
  console.log('  curl -O https://alphafold.ebi.ac.uk/files/AF-P0A7Y4-F1-model_v6.pdb   # -> af.pdb');
  console.log('  curl -O https://alphafold.ebi.ac.uk/files/msa/AF-P0A7Y4-F1-msa_v6.a3m # -> test.a3m');
  process.exit(0);
}

const core = (await import('../gremlin-core.js')).default;
const MSA = (await import('../msa.js')).default;
const { Gremlin } = core;
const { parsePdb, assignSS, cbContacts, solabContacts } = await import('./contacts.mjs');

/* ---- ground truth from the AlphaFold model ---- */
const pdb = parsePdb(readFileSync(DIR + '/af.pdb', 'utf8'));
const nRes = pdb.nRes;
const ss = assignSS(pdb);

/*
 * CB < 8A is the CASP convention and stays the primary metric, so the numbers
 * here remain comparable to everyone else's. The per-SS virtual-Cbeta model
 * from the solab contact page is reported alongside it as a cross-check --
 * see test/contacts.mjs, and the README on how much the choice moves things.
 */
const MINSEP = 5;
const isContact = cbContacts(pdb, 8.0);
const DEFS = [
  ['CB < 8A (CASP)', isContact],
  ['solab per-SS vCB', solabContacts(pdb, ss)]
];
const countTrue = (f) => {
  let n = 0;
  for (let i = 0; i < nRes; i++) for (let j = i + MINSEP; j < nRes; j++) if (f(i, j)) n++;
  return n;
};
console.log('structure: ' + nRes + ' residues');
console.log('secondary structure (simplified DSSP): '
  + [...ss].filter(c => c === 'H').length + ' H, '
  + [...ss].filter(c => c === 'E').length + ' E, '
  + [...ss].filter(c => c === 'L').length + ' L');
for (const [name, f] of DEFS) {
  console.log('true contacts, ' + name.padEnd(18) + ' |i-j|>=' + MINSEP + ': ' + countTrue(f));
}

/* ---- alignment ---- */
const text = readFileSync(DIR + '/test.a3m', 'utf8');
const ds = MSA.buildDataset(text, {
  keepQueryColumns: true, minCoverage: 0.75, minIdentity: 0.15, sortByIdentity: true
});
console.log('msa: N=' + ds.N + ' L=' + ds.L + ' A=' + ds.A);
let identityMap = true;
for (let c = 0; c < ds.L; c++) if (ds.colMap[c] !== c) identityMap = false;
console.log('colMap is identity: ' + identityMap);

const wasm = await core.initWasm(new URL('../gremlin.wasm', import.meta.url).pathname);
console.log('backend: ' + (wasm ? 'wasm' : 'js') + '\n');

function ranking(g) {
  const cm = g.contactMap(), L = g.L, out = [];
  for (let i = 0; i < L; i++) for (let j = i + MINSEP; j < L; j++) out.push([i, j, cm[i * L + j]]);
  out.sort((a, b) => b[2] - a[2]);
  return out;
}
function precisionOf(rank, L, f) {
  const scored = rank.filter(r => f(r[0], r[1]) !== null);
  const at = (k) => {
    const n = Math.min(k, scored.length);
    let hit = 0;
    for (let q = 0; q < n; q++) if (f(scored[q][0], scored[q][1])) hit++;
    return hit / n;
  };
  return { topL: at(L), topL2: at(Math.floor(L / 2)), topL5: at(Math.floor(L / 5)) };
}
function precision(g) { return precisionOf(ranking(g), g.L, isContact); }

const CONFIGS = [
  { name: 'before (none of the four changes)',
    gap: -1, biasInit: 'zero', alpha: 0.02, lr: 0.05 },
  { name: 'all four changes (what ships)',
    gap: 20, biasInit: 'freq', alpha: 0.01, lr: null },
  { name: '  ablate: gaps back in the norm',
    gap: -1, biasInit: 'freq', alpha: 0.01, lr: null },
  { name: '  ablate: bias starts at zero',
    gap: 20, biasInit: 'zero', alpha: 0.01, lr: null },
  { name: '  ablate: old fixed lr 0.05',
    gap: 20, biasInit: 'freq', alpha: 0.01, lr: 0.05 },
  { name: '  ablate: 2x coupling penalty',
    gap: 20, biasInit: 'freq', alpha: 0.02, lr: null }
];

const STEPS = 400, B = 128;
console.log('running ' + STEPS + ' steps at batch ' + B + ' per config\n');
console.log('config                                     lr      top L/5  top L/2  top L');
console.log('-'.repeat(80));

let shipped = null;
for (const c of CONFIGS) {
  const lr = c.lr === null ? Gremlin.suggestLr(ds.L, B) : c.lr;
  const g = new Gremlin({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, backend: wasm,
    gap: c.gap, biasInit: c.biasInit, identity: 0.8, maxRefs: 3000, seed: 1234567,
    cfg: { batch: B, lr, alpha: c.alpha, beta: 0.01, regMode: 'gremlin' }
  });
  for (let s = 0; s < STEPS; s++) g.step();
  const rank = ranking(g);
  if (/what ships/.test(c.name)) shipped = { rank, L: g.L };
  const p = precisionOf(rank, g.L, isContact);
  console.log(c.name.padEnd(42) + lr.toFixed(4).padStart(6) + '   '
    + (p.topL5 * 100).toFixed(1).padStart(6) + '%  '
    + (p.topL2 * 100).toFixed(1).padStart(6) + '%  '
    + (p.topL * 100).toFixed(1).padStart(6) + '%');
}

/*
 * Same predictions, different ground truth. The count-matched CB cutoff is the
 * control: the solab model calls more pairs contacts than CB<8A does, and a
 * more permissive definition raises precision for free, so the only fair
 * comparison is against a plain cutoff tuned to the same number of contacts.
 */
if (shipped) {
  const target = countTrue(solabContacts(pdb, ss));
  let lo = 6, hi = 14, cut;
  for (let it = 0; it < 40; it++) {
    cut = (lo + hi) / 2;
    if (countTrue(cbContacts(pdb, cut)) < target) lo = cut; else hi = cut;
  }
  const ALT = [
    ['CB < 8.0A (CASP, primary)', cbContacts(pdb, 8.0)],
    ['CB < ' + cut.toFixed(1) + 'A (count-matched control)', cbContacts(pdb, cut)],
    ['solab per-SS virtual-CB', solabContacts(pdb, ss)]
  ];
  console.log('\nthe shipped config scored against each ground-truth definition:');
  console.log('ground truth                              n true  top L/5  top L/2  top L');
  console.log('-'.repeat(80));
  for (const [name, f] of ALT) {
    const p = precisionOf(shipped.rank, shipped.L, f);
    console.log(name.padEnd(42) + String(countTrue(f)).padStart(6) + '   '
      + (p.topL5 * 100).toFixed(1).padStart(6) + '%  '
      + (p.topL2 * 100).toFixed(1).padStart(6) + '%  '
      + (p.topL * 100).toFixed(1).padStart(6) + '%');
  }
}
