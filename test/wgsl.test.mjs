/*
 * node test/wgsl.test.mjs
 *
 * Static validation of every WGSL shader in gremlin-gpu.js, using naga (the
 * same front end wgpu uses). WebGPU could not be executed in the environment
 * this was written in, so this is the only compile-time check the GPU path gets:
 * it catches syntax, type, binding and control-flow errors, but NOT a wrong
 * index expression. Numerical correctness is enforced separately, at runtime, by
 * the selfTest gate in gremlin-core.js.
 *
 * Skips (exit 0) if naga is not installed.
 */

import { writeFileSync, mkdtempSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

await import('../gremlin-gpu.js');
const S = globalThis.GremlinGPUShaders;

let naga = null;
for (const cand of ['naga', `${process.env.HOME}/.cargo/bin/naga`]) {
  try { execFileSync(cand, ['--version'], { stdio: 'pipe' }); naga = cand; break; } catch {}
}
if (!naga) {
  console.log('naga not installed -- skipping WGSL validation');
  console.log('  install with: cargo install naga-cli');
  process.exit(0);
}
console.log('naga:', execFileSync(naga, ['--version'], { encoding: 'utf8' }).trim());

const dir = mkdtempSync(join(tmpdir(), 'wgsl-'));
// Every shader the backend compiles. Kept in sync by asserting below that this
// list covers exactly the keys GremlinGPUShaders exports.
const names = ['forward', 'backward', 'biasgrad', 'sym', 'adam', 'adamb', 'frob', 'apc', 'reduce'];

let failed = 0;
for (const n of names) {
  const src = S[n];
  if (!src) { console.log('  FAIL ' + n + ' -- shader missing from GremlinGPUShaders'); failed++; continue; }
  const f = join(dir, n + '.wgsl');
  writeFileSync(f, src);
  try {
    // naga validates on parse; asking for SPIR-V also forces backend lowering,
    // which catches things the front end alone lets through.
    execFileSync(naga, ['--stdin-file-path', f, f, join(dir, n + '.spv')], { stdio: 'pipe' });
    console.log('  ok   ' + n + ' (' + src.split('\n').length + ' lines)');
  } catch (e) {
    const msg = (e.stderr?.toString() || e.stdout?.toString() || e.message).trim();
    console.log('  FAIL ' + n + '\n' + msg.split('\n').map(l => '       ' + l).join('\n'));
    failed++;
  }
}

// Guard against a shader being added to the backend but not to this list.
const exported = Object.keys(S).filter(k => typeof S[k] === 'string');
const missing = exported.filter(k => !names.includes(k));
if (missing.length) {
  console.log('  FAIL shaders exported but not validated: ' + missing.join(', '));
  failed++;
}

// The uniform struct must match what writeCfg packs: 12 x 4 bytes.
const cfgFields = (S.forward.match(/struct Cfg \{([\s\S]*?)\}/) || [, ''])[1]
  .split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('//'));
console.log('\nCfg struct has ' + cfgFields.length + ' fields; writeCfg packs 48 bytes = 12 slots');
if (cfgFields.length !== 12) {
  console.log('  FAIL Cfg field count does not match the 48-byte packing in writeCfg');
  failed++;
}

console.log('\n' + (names.length - failed) + '/' + names.length + ' shaders valid');
process.exit(failed ? 1 : 0);
