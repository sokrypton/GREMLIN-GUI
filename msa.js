/*
 * msa.js -- MSA parsing, filtering and encoding. Shared by the browser UI and
 * the node tests, so no DOM here either.
 *
 * Two input modes, auto-detected:
 *
 *   digit   -- one decimal digit per position, A = max digit + 1. This is the
 *              original toy format and is kept working verbatim.
 *   protein -- FASTA or A3M over the 21-letter alphabet below. This is the mode
 *              the original could not express at all: it stripped everything
 *              except [0-9\n], which caps A at 10 and makes 20 amino acids plus
 *              a gap unrepresentable.
 */
(function (root) {
  'use strict';

  // GREMLIN's conventional ordering, gap last.
  var PROT = 'ARNDCQEGHILKMFPSTWYV-';
  var GAP = 20;
  var A_PROT = 21;

  var PROT_IDX = (function () {
    var m = {}, i;
    for (i = 0; i < PROT.length; i++) m[PROT[i]] = i;
    // everything else -- X B Z J U O * . ~ etc -- folds into the gap state
    return m;
  })();

  // Clustal-ish residue colours, indexed as PROT.
  var PROT_COLORS = [
    '#80a0f0', '#f01505', '#15c015', '#c048c0', '#f08080', '#15c015', '#c048c0',
    '#f09048', '#15a4a4', '#80a0f0', '#80a0f0', '#f01505', '#80a0f0', '#80a0f0',
    '#ffff00', '#15c015', '#15c015', '#80a0f0', '#15a4a4', '#80a0f0', '#f0f0f0'
  ];

  // The original toy palette, for digit mode.
  var DIGIT_COLORS = [
    '#eab308', '#3b82f6', '#22c55e', '#ef4444', '#8b5cf6',
    '#ec4899', '#f97316', '#10b981', '#6366f1', '#f43f5e'
  ];

  /* ------------------------------------------------------------------ */
  /* py2Dmol-compatible helpers                                          */
  /* ------------------------------------------------------------------ */

  /* AlphaFold DB serves the a3m it used for each prediction. CORS is '*', so a
     static page can fetch it directly; 404 means no MSA for that accession. */
  function afdbMsaUrl(accession) {
    return 'https://alphafold.ebi.ac.uk/files/msa/AF-'
      + String(accession).trim().toUpperCase() + '-F1-msa_v6.a3m';
  }

  // Matches py2Dmol's isGapResidue: unknown residues count as gaps for the
  // purpose of coverage and identity.
  function isGap(ch) {
    return !ch || ch === '-' || ch === '.' || ch === ' ' || ch === 'X';
  }

  /** Fraction of aligned columns where this sequence has a real residue. */
  function coverageOf(seq) {
    if (!seq || !seq.length) return 0;
    var n = 0;
    for (var i = 0; i < seq.length; i++) if (!isGap(seq[i])) n++;
    return n / seq.length;
  }

  /** Fraction of columns where both sequences have the same real residue. */
  function identityTo(seq, query) {
    if (!seq || !query || seq.length !== query.length) return 0;
    var m = 0, t = 0, a, b;
    for (var i = 0; i < seq.length; i++) {
      a = seq[i]; b = query[i];
      t++;
      if (!isGap(a) && !isGap(b) && a === b) m++;
    }
    return t > 0 ? m / t : 0;
  }

  /* ------------------------------------------------------------------ */
  /* redundancy filter                                                   */
  /* ------------------------------------------------------------------ */

  /*
   * Greedy clustering at a pairwise identity threshold: walk the sequences in
   * order, keep one as a representative, and drop any later sequence that is
   * more than `maxId` identical to a representative already kept. Row 0 (the
   * query) is always a representative. Order matters, and after sortByIdentity
   * it is descending identity to the query, so the representative kept from a
   * cluster is the one most like the query.
   *
   * Identity here is "fraction of the L aligned columns that are equal",
   * counting gap-against-gap as a match -- the same definition computeWeights
   * uses for Meff, and the one the GREMLIN reference uses.
   *
   * Comparison is byte-packed, four residues per 32-bit word, with an early exit
   * once more than L - need mismatches have accumulated.
   *
   * ---------------------------------------------------------------------------
   * On prefiltering, which was tried and removed
   * ---------------------------------------------------------------------------
   * Because every sequence occupies the same columns, the match count is bounded
   * above by  sum_a min(count_1[a], count_2[a])  -- each residue type can only
   * match as often as the rarer of the two sequences contains it. That bound is
   * exact and screens beautifully: on a 15688 x 155 alignment it prunes 95.9% of
   * pairs at id 0.90.
   *
   * It still made things slower. Measured, exact-only against bound-then-exact:
   *
   *     id 0.90   9.74s -> 9.92s   (0.98x, 95.9% pruned)
   *     id 0.80  10.51s -> 12.53s  (0.84x, 27.2% pruned)
   *     id 0.70   3.47s ->  4.93s  (0.70x,  2.6% pruned)
   *
   * The reason is that the test being screened is already cheaper than the
   * screen. At id 0.90 the exact comparison bails after L - need + 1 = 16
   * mismatches, roughly five word-comparisons, while the bound costs up to 21
   * integer mins. A one-operation variant -- matches <= L - |gaps_1 - gaps_2|,
   * also exact -- is cheap enough but only prunes 6%, and was likewise a wash.
   *
   * If this ever needs to be faster the answer is an inverted k-mer index over
   * the representatives (CD-HIT's approach), not a tighter per-pair bound.
   *
   * The same measurement was repeated against the Meff pass, where the bound
   * prunes less (21.8% at threshold 0.8) and costs more (0.70x). Same reason,
   * same verdict.
   *
   * ---------------------------------------------------------------------------
   * Relation to Meff reweighting
   * ---------------------------------------------------------------------------
   * This is the same clustering computeWeights does in its 'cluster' weight
   * mode -- one greedy pass, each sequence compared against representatives
   * only. The two differ purely in what they do with the partition:
   *
   *     this           keep one member per cluster, drop the rest      N -> #clusters
   *     computeWeights keep every sequence at weight 1/|its cluster|   Meff = #clusters
   *
   * Both land on the same Meff, which is why filtering at a threshold makes
   * reweighting at that same threshold a no-op -- see the `filteredAt` skip in
   * computeWeights. Reweighting is the better default of the two, since it
   * reaches the same Meff without discarding sequences.
   *
   * Returns { keep: Int32Array, stats }.
   */
  function filterRedundancy(seqs, N, L, A, maxId, onProgress) {
    var need = Math.ceil(maxId * L);           // matches required to count as redundant
    var n, m, k;

    var words = (L / 4) | 0;
    var tail = L - words * 4;
    var stride = words * 4 + (tail ? 4 : 0);
    var bytes = new Uint8Array(N * stride);
    for (n = 0; n < N; n++) {
      for (k = 0; k < L; k++) bytes[n * stride + k] = seqs[n * L + k];
    }
    var u32 = new Uint32Array(bytes.buffer);
    var wStride = stride >> 2;

    function identical(an, am) {
      var id = 0, w, x, remaining;
      for (w = 0; w < words; w++) {
        x = u32[an + w] ^ u32[am + w];
        if (x === 0) { id += 4; continue; }
        if ((x & 0xff) === 0) id++;
        if ((x & 0xff00) === 0) id++;
        if ((x & 0xff0000) === 0) id++;
        if ((x & 0xff000000) === 0) id++;
        remaining = L - (w + 1) * 4;
        if (id + remaining < need) return false;
      }
      if (tail) {
        x = u32[an + words] ^ u32[am + words];
        if ((tail > 0) && (x & 0xff) === 0) id++;
        if ((tail > 1) && (x & 0xff00) === 0) id++;
        if ((tail > 2) && (x & 0xff0000) === 0) id++;
      }
      return id >= need;
    }

    var keep = [], reps = [];
    var compared = 0;
    for (n = 0; n < N; n++) {
      var dup = false;
      for (var r = 0; r < reps.length; r++) {
        compared++;
        if (identical(n * wStride, reps[r] * wStride)) { dup = true; break; }
      }
      if (!dup) { reps.push(n); keep.push(n); }
      if (onProgress && (n & 255) === 0) onProgress(n / N);
    }

    return {
      keep: Int32Array.from(keep),
      stats: { kept: keep.length, dropped: N - keep.length, compared: compared }
    };
  }

  /* ------------------------------------------------------------------ */
  /* parsing                                                            */
  /* ------------------------------------------------------------------ */

  /** Split FASTA/A3M, or fall back to one sequence per non-blank line. */
  function parseRecords(text) {
    var names = [], seqs = [];
    if (/^\s*>/.test(text)) {
      var lines = text.split(/\r?\n/), cur = null, i, ln;
      for (i = 0; i < lines.length; i++) {
        ln = lines[i];
        if (ln.charCodeAt(0) === 62) {                     // '>'
          names.push(ln.slice(1).trim());
          cur = [];
          seqs.push(cur);
        } else if (cur && ln.length) {
          cur.push(ln.trim());
        }
      }
      seqs = seqs.map(function (parts) { return parts.join(''); });
    } else {
      text.split(/\r?\n/).forEach(function (ln, i) {
        var s = ln.trim();
        if (s.length) { names.push('seq' + i); seqs.push(s); }
      });
    }
    var keep = [];
    for (var k = 0; k < seqs.length; k++) if (seqs[k].length) keep.push(k);
    return { names: keep.map(function (k) { return names[k]; }),
             seqs: keep.map(function (k) { return seqs[k]; }) };
  }

  function isDigitMode(seqs) {
    for (var i = 0; i < seqs.length; i++) if (!/^[0-9]*$/.test(seqs[i])) return false;
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* buildDataset                                                       */
  /* ------------------------------------------------------------------ */

  /**
   * opts = {
   *   keepQueryColumns : drop columns where the first sequence has a gap (default true)
   *   maxColGap        : drop columns whose gap fraction exceeds this (default 1 = off)
   *   maxSeqGap        : drop sequences whose gap fraction exceeds this (default 0.5)
   *   dedup            : collapse exact duplicate sequences (default false --
   *                      it is not equivalent to reweighting, since duplicates
   *                      would otherwise each carry 1/|cluster|)
   * }
   *
   * Returns { mode, L, A, N, seqs: Int32Array(N*L), colMap, names, symbols,
   *           colors, warnings, raw:{N,L} }
   * colMap[c] is the column index in the *input* alignment, so contacts can be
   * reported in query numbering after filtering.
   */
  function buildDataset(text, opts) {
    opts = opts || {};
    var keepQuery = opts.keepQueryColumns !== false;
    var maxColGap = opts.maxColGap === undefined ? 1 : opts.maxColGap;
    // py2Dmol's thresholds and defaults: coverage is the non-gap fraction of a
    // row, identity is measured against the query (row 0).
    var minCoverage = opts.minCoverage === undefined ? 0 : opts.minCoverage;
    var minIdentity = opts.minIdentity === undefined ? 0 : opts.minIdentity;
    var sortByIdentity = !!opts.sortByIdentity;
    // Redundancy filter: drop sequences more than this identical to one already
    // kept. 1 (or 0) disables it. See filterRedundancy.
    var maxIdentity = opts.maxIdentity === undefined ? 1 : opts.maxIdentity;
    var warnings = [];

    var rec = parseRecords(text);
    var names = rec.names, raw = rec.seqs;
    if (!raw.length) throw new Error('No sequences found.');

    var digit = isDigitMode(raw);
    var A, symbols, colors, toIdx, gap;

    if (digit) {
      var mx = 1, i, j, c;
      for (i = 0; i < raw.length; i++) {
        for (j = 0; j < raw[i].length; j++) {
          c = raw[i].charCodeAt(j) - 48;
          if (c > mx) mx = c;
        }
      }
      A = mx + 1;
      symbols = [];
      for (i = 0; i < A; i++) symbols.push(String(i));
      colors = DIGIT_COLORS.slice(0, A);
      gap = -1;                                            // digit mode has no gap state
      toIdx = function (ch) { var v = ch.charCodeAt(0) - 48; return v >= 0 && v < A ? v : 0; };
    } else {
      // A3M: lowercase letters (and '.') are insertions relative to the query.
      // Only strip them if the raw lengths actually disagree -- some plain
      // FASTA alignments use lowercase for soft-masking, not insertions.
      var lens = raw.map(function (s) { return s.length; });
      var uneven = Math.min.apply(null, lens) !== Math.max.apply(null, lens);
      var hasLower = raw.some(function (s) { return /[a-z.]/.test(s); });
      if (uneven && hasLower) {
        raw = raw.map(function (s) { return s.replace(/[a-z.]/g, ''); });
        warnings.push('Treated lowercase columns as A3M insertions and removed them.');
      }
      raw = raw.map(function (s) { return s.toUpperCase().replace(/\./g, '-'); });
      A = A_PROT;
      symbols = PROT.split('');
      colors = PROT_COLORS.slice();
      gap = GAP;
      toIdx = function (ch) { var v = PROT_IDX[ch]; return v === undefined ? GAP : v; };
    }

    // Ragged input: pad rather than reject, but say so.
    var Lraw = 0, n;
    for (n = 0; n < raw.length; n++) if (raw[n].length > Lraw) Lraw = raw[n].length;
    var padChar = digit ? '0' : '-';
    var padded = 0;
    for (n = 0; n < raw.length; n++) {
      if (raw[n].length < Lraw) { raw[n] = raw[n] + padChar.repeat(Lraw - raw[n].length); padded++; }
    }
    if (padded) warnings.push(padded + ' sequence(s) were shorter than the alignment and were padded.');

    var rawN = raw.length, rawL = Lraw;

    /* ---- column selection ---- */
    var cols = [];
    var c2, gapCount, want;
    for (c2 = 0; c2 < Lraw; c2++) {
      if (!digit) {
        if (keepQuery && raw[0][c2] === '-') continue;
        if (maxColGap < 1) {
          gapCount = 0;
          for (n = 0; n < rawN; n++) if (raw[n][c2] === '-') gapCount++;
          if (gapCount / rawN > maxColGap) continue;
        }
      }
      cols.push(c2);
    }
    if (!cols.length) throw new Error('Column filters removed every column.');
    if (cols.length < Lraw) {
      warnings.push('Kept ' + cols.length + ' of ' + Lraw + ' columns.');
    }

    /* ---- sequence selection (py2Dmol pipeline: coverage, then identity) ---- */
    // Project each row onto the kept columns once; coverage, identity, sorting,
    // dedup and encoding all read the projection instead of re-indexing raw.
    var proj = new Array(rawN), pj;
    for (n = 0; n < rawN; n++) {
      pj = '';
      for (c2 = 0; c2 < cols.length; c2++) pj += raw[n][cols[c2]];
      proj[n] = pj;
    }
    var query = proj[0];

    var keepSeq = [], cov, idn;
    var covs = new Float32Array(rawN), idns = new Float32Array(rawN);
    var nDropCov = 0, nDropId = 0;
    for (n = 0; n < rawN; n++) {
      cov = digit ? 1 : coverageOf(proj[n]);
      idn = digit ? 1 : (n === 0 ? 1 : identityTo(proj[n], query));
      covs[n] = cov; idns[n] = idn;
      if (n > 0) {                                   // the query is never dropped
        if (cov < minCoverage) { nDropCov++; continue; }
        if (idn < minIdentity) { nDropId++; continue; }
      }
      keepSeq.push(n);
    }
    if (!keepSeq.length) throw new Error('Sequence filters removed every sequence.');
    if (nDropCov) warnings.push('Dropped ' + nDropCov + ' sequence(s) below ' + minCoverage.toFixed(2) + ' coverage.');
    if (nDropId) warnings.push('Dropped ' + nDropId + ' sequence(s) below ' + minIdentity.toFixed(2) + ' identity to the query.');

    if (sortByIdentity && !digit) {
      // descending identity, query pinned first -- py2Dmol's display order
      keepSeq.sort(function (p, q) {
        if (p === 0) return -1;
        if (q === 0) return 1;
        return idns[q] - idns[p];
      });
    }

    if (opts.dedup) {
      var seen = Object.create(null), ded = [];
      for (n = 0; n < keepSeq.length; n++) {
        if (seen[proj[keepSeq[n]]] === undefined) { seen[proj[keepSeq[n]]] = 1; ded.push(keepSeq[n]); }
      }
      if (ded.length < keepSeq.length) {
        warnings.push('Removed ' + (keepSeq.length - ded.length) + ' exact duplicate sequence(s).');
      }
      keepSeq = ded;
    }

    /* ---- encode ---- */
    var L = cols.length, N = keepSeq.length;
    var seqs = new Int32Array(N * L);
    for (n = 0; n < N; n++) {
      var src = proj[keepSeq[n]], off = n * L;
      for (c2 = 0; c2 < L; c2++) seqs[off + c2] = toIdx(src[c2]);
    }

    // In digit mode A was derived from the whole input; recompute over what
    // survived filtering so the model is not sized for absent states.
    if (digit) {
      var mx2 = 1;
      for (var k = 0; k < seqs.length; k++) if (seqs[k] > mx2) mx2 = seqs[k];
      if (mx2 + 1 < A) {
        A = mx2 + 1;
        symbols = symbols.slice(0, A);
        colors = colors.slice(0, A);
      }
    }

    /* ---- redundancy filter, on the encoded sequences ---- */
    var redundancy = null;
    if (maxIdentity > 0 && maxIdentity < 1 && N > 1) {
      var fr = filterRedundancy(seqs, N, L, A, maxIdentity, opts.onProgress);
      redundancy = fr.stats;
      if (fr.keep.length < N) {
        var kept = new Int32Array(fr.keep.length * L);
        for (n = 0; n < fr.keep.length; n++) {
          kept.set(seqs.subarray(fr.keep[n] * L, (fr.keep[n] + 1) * L), n * L);
        }
        keepSeq = Array.prototype.map.call(fr.keep, function (k) { return keepSeq[k]; });
        seqs = kept;
        N = fr.keep.length;
        warnings.push('Removed ' + fr.stats.dropped + ' sequence(s) above '
          + maxIdentity.toFixed(2) + ' identity to a kept sequence; ' + N + ' remain.');
      }
    }

    return {
      mode: digit ? 'digit' : 'protein',
      L: L, A: A, N: N,
      seqs: seqs,
      redundancy: redundancy,
      colMap: Int32Array.from(cols),
      names: keepSeq.map(function (k) { return names[k] || ('seq' + k); }),
      cov: Float32Array.from(keepSeq, function (k) { return covs[k]; }),
      idn: Float32Array.from(keepSeq, function (k) { return idns[k]; }),
      // deliberately no decoded string rows: at N=100k that is tens of MB of
      // JS strings for something the canvas viewer reads straight out of seqs
      symbols: symbols,
      colors: colors,
      gap: gap,
      warnings: warnings,
      raw: { N: rawN, L: rawL }
    };
  }

  /* ------------------------------------------------------------------ */
  /* synthetic MSA with planted contacts                                */
  /* ------------------------------------------------------------------ */

  /*
   * Generates an alignment whose only real structure is a known set of coupled
   * position pairs, so contact recovery can be checked without a reference
   * structure. Used by the "Synthetic" button in the UI and by the node tests.
   *
   * Each planted pair (i,j) draws a state at i and sets j to a fixed partner
   * state with probability `strength`; unpaired positions draw from a
   * position-specific biased distribution, which gives the bias term something
   * to explain and keeps the couplings from absorbing single-site signal.
   */
  function synthetic(opts) {
    opts = opts || {};
    var L = opts.L || 48;
    var N = opts.N || 800;
    // Emit real residues only -- never the gap state, or the query-gap column
    // filter would drop columns and renumber the planted pairs.
    var A = Math.min(opts.A || 20, A_PROT - 1);
    var nPairs = opts.nPairs || Math.max(1, Math.floor(L / 6));
    var minSep = opts.minSep || 4;
    var strength = opts.strength === undefined ? 0.95 : opts.strength;
    var st = (opts.seed | 0) || 20260804;

    function rnd() {
      st ^= st << 13; st >>>= 0;
      st ^= st >>> 17;
      st ^= st << 5; st >>>= 0;
      return st / 4294967296;
    }

    // disjoint pairs, separated by at least minSep
    var used = new Uint8Array(L), pairs = [], tries = 0, i, j;
    while (pairs.length < nPairs && tries++ < 4000) {
      i = (rnd() * L) | 0; j = (rnd() * L) | 0;
      if (i === j || used[i] || used[j] || Math.abs(i - j) < minSep) continue;
      used[i] = used[j] = 1;
      pairs.push(i < j ? [i, j] : [j, i]);
    }

    // per-position preference over states, so single-site frequencies are not flat
    var bias = [];
    for (i = 0; i < L; i++) {
      var w = new Float64Array(A), tot = 0, a;
      for (a = 0; a < A; a++) { w[a] = Math.pow(rnd(), 3) + 0.02; tot += w[a]; }
      for (a = 0; a < A; a++) w[a] /= tot;
      bias.push(w);
    }
    function draw(p) {
      var r = rnd(), acc = 0;
      for (var a = 0; a < p.length; a++) { acc += p[a]; if (r <= acc) return a; }
      return p.length - 1;
    }

    var partner = pairs.map(function () { return (rnd() * A) | 0; });
    var alphabet = PROT.slice(0, A);
    var seqs = [], names = [];

    for (var n = 0; n < N; n++) {
      var x = new Int32Array(L);
      for (i = 0; i < L; i++) x[i] = draw(bias[i]);
      for (var p = 0; p < pairs.length; p++) {
        var a0 = x[pairs[p][0]];
        // j is a deterministic function of i's state most of the time
        x[pairs[p][1]] = rnd() < strength ? (a0 + partner[p]) % A : draw(bias[pairs[p][1]]);
      }
      var s = '';
      for (i = 0; i < L; i++) s += alphabet[x[i]];
      seqs.push(s);
      names.push('sim' + n);
    }

    var text = '';
    for (var q = 0; q < seqs.length; q++) text += '>' + names[q] + '\n' + seqs[q] + '\n';
    return { text: text, seqs: seqs, names: names, pairs: pairs, L: L, N: N, A: A };
  }

  root.MSA = {
    PROT: PROT,
    GAP: GAP,
    A_PROT: A_PROT,
    PROT_COLORS: PROT_COLORS,
    DIGIT_COLORS: DIGIT_COLORS,
    parseRecords: parseRecords,
    isDigitMode: isDigitMode,
    buildDataset: buildDataset,
    filterRedundancy: filterRedundancy,
    synthetic: synthetic,
    afdbMsaUrl: afdbMsaUrl,
    isGap: isGap,
    coverageOf: coverageOf,
    identityTo: identityTo
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.MSA;
})(typeof globalThis !== 'undefined' ? globalThis : this);
