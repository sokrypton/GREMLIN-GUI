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

/* ---- ground truth from the AlphaFold model ---- */
const pdb = readFileSync(DIR + '/af.pdb', 'utf8').split('\n');
const coord = new Map();          // resSeq -> [x,y,z]
for (const ln of pdb) {
  if (!ln.startsWith('ATOM')) continue;
  const atom = ln.slice(12, 16).trim();
  const resn = ln.slice(17, 20).trim();
  const seq = parseInt(ln.slice(22, 26), 10);
  const want = (resn === 'GLY') ? 'CA' : 'CB';
  if (atom !== want) continue;
  coord.set(seq, [parseFloat(ln.slice(30, 38)), parseFloat(ln.slice(38, 46)), parseFloat(ln.slice(46, 54))]);
}
const nRes = Math.max(...coord.keys());
console.log('structure: ' + coord.size + ' residues with a CB/CA, max resSeq ' + nRes);

const MINSEP = 5, CUT = 8.0;
function isContact(i, j) {                  // i, j are 0-based model columns
  const a = coord.get(i + 1), b = coord.get(j + 1);
  if (!a || !b) return null;
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz) < CUT;
}
let nTrue = 0;
for (let i = 0; i < nRes; i++) for (let j = i + MINSEP; j < nRes; j++) if (isContact(i, j)) nTrue++;
console.log('true contacts (CB<8A, |i-j|>=5): ' + nTrue);

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

function precision(g, steps) {
  const cm = g.contactMap();
  const L = g.L, out = [];
  for (let i = 0; i < L; i++) {
    for (let j = i + MINSEP; j < L; j++) {
      const t = isContact(i, j);
      if (t !== null) out.push([i, j, cm[i * L + j], t]);
    }
  }
  out.sort((a, b) => b[2] - a[2]);
  const at = (k) => {
    const n = Math.min(k, out.length);
    let hit = 0;
    for (let q = 0; q < n; q++) if (out[q][3]) hit++;
    return hit / n;
  };
  return { topL: at(L), topL2: at(Math.floor(L / 2)), topL5: at(Math.floor(L / 5)) };
}

const CONFIGS = [
  { name: 'before (this branch, pre-reference)',
    gap: -1, biasInit: 'zero', alpha: 0.02, lr: 0.05 },
  { name: 'reference-aligned (all four changes)',
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

for (const c of CONFIGS) {
  const lr = c.lr === null ? Gremlin.suggestLr(ds.L, B) : c.lr;
  const g = new Gremlin({
    L: ds.L, A: ds.A, N: ds.N, seqs: ds.seqs, backend: wasm,
    gap: c.gap, biasInit: c.biasInit, identity: 0.8, maxRefs: 3000, seed: 1234567,
    cfg: { batch: B, lr, alpha: c.alpha, beta: 0.01, regMode: 'gremlin' }
  });
  for (let s = 0; s < STEPS; s++) g.step();
  const p = precision(g, STEPS);
  console.log(c.name.padEnd(42) + lr.toFixed(4).padStart(6) + '   '
    + (p.topL5 * 100).toFixed(1).padStart(6) + '%  '
    + (p.topL2 * 100).toFixed(1).padStart(6) + '%  '
    + (p.topL * 100).toFixed(1).padStart(6) + '%');
}
