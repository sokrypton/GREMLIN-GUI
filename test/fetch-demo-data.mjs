/*
 * node test/fetch-demo-data.mjs [dir]
 *
 * Writes the two inputs eval-precision.mjs needs for the demo protein, E. coli
 * RNase HI (UniProt P0A7Y4, 155 residues):
 *
 *   test.a3m  built by the ColabFold MMseqs2 server, exactly as the practical
 *             page's Demo button builds it (mmseqs.js)
 *   af.pdb    the crystal structure 2RN2 from RCSB, residues 1-155 -- the same
 *             numbering as the UniProt sequence
 *
 * These replace the AlphaFold DB model and a3m the README's older numbers were
 * measured with: AFDB's /files/ path refuses requests since Nov 2026.
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const MMseqs = require('../mmseqs.js');
const dir = process.argv[2] || '.';
mkdirSync(dir, { recursive: true });

const pdb = await fetch('https://files.rcsb.org/download/2RN2.pdb');
if (!pdb.ok) throw new Error('RCSB returned HTTP ' + pdb.status);
writeFileSync(dir + '/af.pdb', await pdb.text());
console.log('wrote ' + dir + '/af.pdb (2RN2)');

const seq = await MMseqs.uniprotSequence('P0A7Y4');
const r = await MMseqs.search(seq, { onProgress: (p) => console.log('  mmseqs2: ' + p.phase) });
writeFileSync(dir + '/test.a3m', r.a3m);
console.log('wrote ' + dir + '/test.a3m (' + (r.a3m.match(/^>/gm) || []).length + ' sequences)');
