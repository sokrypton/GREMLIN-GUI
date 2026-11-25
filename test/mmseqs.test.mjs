/*
 * node test/mmseqs.test.mjs
 *
 * The offline half of mmseqs.js: reading the result archive and turning it into
 * one alignment. The network half is exercised by hand against the live server
 * (it is shared and rate-limited, so tests do not call it).
 */
import assert from 'node:assert/strict';

const M = (await import('../mmseqs.js')).default;
const MSA = (await import('../msa.js')).default;

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log('  ok   ' + name); passed++; }
  catch (e) { console.log('  FAIL ' + name + '\n       ' + e.message); failed++; }
}

/* A minimal POSIX tar: 512-byte header, data padded to 512, two zero blocks. */
function tar(files) {
  const enc = new TextEncoder(), parts = [];
  for (const [name, text] of Object.entries(files)) {
    const data = enc.encode(text), h = new Uint8Array(512);
    h.set(enc.encode(name), 0);
    h.set(enc.encode(data.length.toString(8).padStart(11, '0') + '\0'), 124);
    h[156] = 48;                                         // '0': regular file
    parts.push(h, data, new Uint8Array((512 - data.length % 512) % 512));
  }
  parts.push(new Uint8Array(1024));
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

const Q = 'MKTAYIAKQR';
const uniref = '>102\nAAAA\n>x\nAAAA\0>101\n' + Q + '\n>hitA\nMKTAYIAKQK\n>hitB\nMKT-YIAKQR\n';
const env = '>101\n' + Q + '\n>envA\nMKSAYIAKQR\n';

test('reads the regular files of a tar', () => {
  const f = M.readTar(tar({ 'uniref.a3m': 'abc', './x/y.a3m': 'de' }));
  assert.deepEqual(Object.keys(f).sort(), ['uniref.a3m', 'x/y.a3m']);
  assert.equal(new TextDecoder().decode(f['uniref.a3m']), 'abc');
});

test('picks the query block and appends env hits without repeating the query', () => {
  const a3m = M.a3mFromTar(tar({ 'uniref.a3m': uniref, 'bfd.mgnify30.metaeuk30.smag30.a3m': env }));
  const names = a3m.match(/^>\S+/gm);
  assert.deepEqual(names, ['>101', '>hitA', '>hitB', '>envA']);
  const ds = MSA.buildDataset(a3m, { minCoverage: 0, minIdentity: 0 });
  assert.equal(ds.L, Q.length);
});

test('works without the environmental file', () => {
  const a3m = M.a3mFromTar(tar({ 'uniref.a3m': uniref }));
  assert.deepEqual(a3m.match(/^>\S+/gm), ['>101', '>hitA', '>hitB']);
});

test('a missing uniref.a3m is an error, not an empty alignment', () => {
  assert.throws(() => M.a3mFromTar(tar({ 'other.a3m': env })), /uniref/);
});

test('tells accessions from sequences', () => {
  assert.equal(M.looksLikeSequence('P0A7Y4'), false);
  assert.equal(M.looksLikeSequence('A0A024RBG1'), false);
  assert.equal(M.looksLikeSequence('MLKQVEIFTDGSCLGNPGPGGYGAIL'), true);
  assert.equal(M.looksLikeSequence('>q\nMLKQ'), true);
});

test('normalize strips FASTA headers, whitespace and digits; rejects junk', () => {
  assert.equal(M.normalize('>q desc\n10 mlkq veif\n20 tdgs*'), 'MLKQVEIFTDGS');
  assert.throws(() => M.normalize('MLKQ-VEIF'), /protein sequence/);
  assert.throws(() => M.normalize(''), /Enter/);
});

console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
