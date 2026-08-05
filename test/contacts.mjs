/*
 * Ground-truth contact definitions for scoring a predicted contact map, plus
 * the secondary-structure assignment one of them needs.
 *
 * Two definitions live here because the choice is not obvious and it moves the
 * reported numbers:
 *
 *   cbContacts     the CASP convention -- Cbeta (Calpha for Gly) within 8 A.
 *                  One cutoff, no structural context. This is what the
 *                  precision numbers in the README were originally measured
 *                  against.
 *
 *   solabContacts  the per-SS virtual-Cbeta model from the solab contact page
 *                  (sokrypton/solab, assets/js/contact.js). A residue-residue
 *                  interaction needs the side chains pointing at each other,
 *                  not just backbone proximity, so each side chain is
 *                  approximated by a point projected off the Calpha trace --
 *                  along the bisector of the two Calpha-Calpha bonds, away from
 *                  the backbone -- at a per-secondary-structure offset, and the
 *                  cutoff is also per-SS. Calibrated there against ConFind
 *                  contact degree > 0.01 over 151 native domains, where it
 *                  scored F1 0.78 against ConFind versus ~0.3 for a raw Calpha
 *                  cutoff.
 *
 * Note the offsets (3.0-4.0 A) are much longer than a real Calpha-Cbeta bond
 * (~1.53 A): the point is a side-chain centroid proxy, not a Cbeta, which is
 * also why it depends on secondary structure -- helix and strand side chains
 * project differently.
 */

/* per-SS virtual-Cbeta model, values as fitted in sokrypton/solab */
export const CBOFF = { H: 3.0, E: 4.0, L: 3.5 };
export function CBCUT(a, b) { return (a === 'H' && b === 'H') ? 8.0 : 8.5; }

/* ------------------------------------------------------------------ */
/* PDB                                                                */
/* ------------------------------------------------------------------ */

/**
 * Backbone + Cbeta per residue, indexed by residue sequence number.
 * Returns { nRes, atom(resSeq, name) -> [x,y,z] | null, resName(resSeq) }.
 */
export function parsePdb(text) {
  const byRes = new Map();
  for (const ln of text.split('\n')) {
    if (!ln.startsWith('ATOM')) continue;
    const name = ln.slice(12, 16).trim();
    const resn = ln.slice(17, 20).trim();
    const seq = parseInt(ln.slice(22, 26), 10);
    let r = byRes.get(seq);
    if (!r) { r = { resn, atoms: {} }; byRes.set(seq, r); }
    r.atoms[name] = [parseFloat(ln.slice(30, 38)),
                     parseFloat(ln.slice(38, 46)),
                     parseFloat(ln.slice(46, 54))];
  }
  const nRes = Math.max(...byRes.keys());
  return {
    nRes,
    atom: (seq, name) => { const r = byRes.get(seq); return (r && r.atoms[name]) || null; },
    resName: (seq) => { const r = byRes.get(seq); return r ? r.resn : null; }
  };
}

/* ------------------------------------------------------------------ */
/* secondary structure                                                */
/* ------------------------------------------------------------------ */

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const len = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
const norm = (a) => { const l = len(a) || 1; return scale(a, 1 / l); };
const dist = (a, b) => len(sub(a, b));

/*
 * Secondary structure, Calpha only, ported from sokrypton/CIRPIN-web
 * (src/tmalign.js makeSec / smoothSec), itself a port of TM-align's make_sec
 * (TMalign.cpp:2466) checked against the C++ to 5e-11.
 *
 * Calpha-only is the right choice here rather than a fallback: the contact
 * model it feeds is itself built off the Calpha trace, so the two agree about
 * what information they are allowed to use. It classifies a residue by how the
 * distances among its five-residue neighbourhood compare with idealised helix
 * and strand geometry.
 */
function secStr(dis13, dis14, dis15, dis24, dis25, dis35) {
  let delta = 2.1;
  if (Math.abs(dis15 - 6.37) < delta && Math.abs(dis14 - 5.18) < delta
    && Math.abs(dis25 - 5.18) < delta && Math.abs(dis13 - 5.45) < delta
    && Math.abs(dis24 - 5.45) < delta && Math.abs(dis35 - 5.45) < delta) return 'H';
  delta = 1.42;
  if (Math.abs(dis15 - 13) < delta && Math.abs(dis14 - 10.4) < delta
    && Math.abs(dis25 - 10.4) < delta && Math.abs(dis13 - 6.1) < delta
    && Math.abs(dis24 - 6.1) < delta && Math.abs(dis35 - 6.1) < delta) return 'E';
  if (dis15 < 8) return 'T';
  return 'C';
}

/** TM-align make_sec over an array of Calpha [x,y,z]. Emits H / E / T / C. */
export function makeSec(ca) {
  const n = ca.length, sec = new Array(n).fill('C');
  for (let i = 2; i + 2 < n; i++) {
    const p = [ca[i - 2], ca[i - 1], ca[i], ca[i + 1], ca[i + 2]];
    if (p.some(q => !q)) continue;
    sec[i] = secStr(dist(p[0], p[2]), dist(p[0], p[3]), dist(p[0], p[4]),
                    dist(p[1], p[3]), dist(p[1], p[4]), dist(p[2], p[4]));
  }
  return sec.join('');
}

/*
 * make_sec is deliberately conservative -- it needs five consecutive Calpha
 * matching idealised geometry, so it marks element cores and leaves the ends
 * coil. CIRPIN-web's smoothSec does the two defensible tidies (bridge a
 * one-residue gap inside an element; drop a lone element residue) without
 * growing element ends, which would invent structure that was never assigned.
 */
export function smoothSec(sec) {
  const a = [...sec];
  for (let i = 0; i + 2 < a.length; i++) {
    for (const j of ['H', 'E']) {
      if (a[i] === j && a[i + 1] !== j && a[i + 2] === j) a[i + 1] = j;
    }
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== 'H' && a[i] !== 'E') continue;
    const before = i > 0 && a[i - 1] === a[i];
    const after = i + 1 < a.length && a[i + 1] === a[i];
    if (!before && !after) a[i] = 'C';
  }
  return a.join('');
}

/**
 * Per-residue H / E / L for the contact model. TM-align's turn and coil states
 * both become L, which is the loop offset CBOFF carries.
 */
export function assignSS(pdb) {
  const ca = [];
  for (let i = 1; i <= pdb.nRes; i++) ca.push(pdb.atom(i, 'CA'));
  return smoothSec(makeSec(ca)).replace(/[TC]/g, 'L');
}

/* ------------------------------------------------------------------ */
/* contact definitions                                                */
/* ------------------------------------------------------------------ */

/**
 * CASP convention: Cbeta (Calpha for Gly) within `cut` angstrom.
 * Returns contact(i, j) over 0-based residue indices, or null if either residue
 * has no coordinates.
 */
export function cbContacts(pdb, cut = 8.0) {
  const p = [];
  for (let i = 1; i <= pdb.nRes; i++) {
    const want = pdb.resName(i) === 'GLY' ? 'CA' : 'CB';
    p.push(pdb.atom(i, want) || pdb.atom(i, 'CA'));
  }
  return (i, j) => {
    const a = p[i], b = p[j];
    if (!a || !b) return null;
    return dist(a, b) < cut;
  };
}

/**
 * The solab per-SS virtual-Cbeta model. `ss` is a per-residue H/E/L string.
 *
 * The virtual point is built from the Calpha trace alone, exactly as on the
 * contact page: sum the vectors to the two Calpha neighbours, normalise, and
 * step CBOFF[ss] along it. That direction points away from the backbone, so it
 * stands in for where the side chain goes.
 */
export function solabContacts(pdb, ss) {
  const n = pdb.nRes;
  const CA = [];
  for (let i = 1; i <= n; i++) CA.push(pdb.atom(i, 'CA'));
  const CB = [];
  for (let i = 0; i < n; i++) {
    if (!CA[i]) { CB.push(null); continue; }
    let v = [0, 0, 0];
    if (i > 0 && CA[i - 1]) v = add(v, sub(CA[i], CA[i - 1]));
    if (i < n - 1 && CA[i + 1]) v = add(v, sub(CA[i], CA[i + 1]));
    const d = len(v) > 1e-6 ? norm(v) : [0, 0, 1];
    CB.push(add(CA[i], scale(d, CBOFF[ss[i]] || 3.5)));
  }
  return (i, j) => {
    const a = CB[i], b = CB[j];
    if (!a || !b) return null;
    return dist(a, b) < CBCUT(ss[i], ss[j]);
  };
}
