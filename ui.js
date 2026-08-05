/*
 * ui.js -- drawing and formatting shared by the two pages.
 *
 * index.html    (edu.js)       the original educational visualization
 * practical.html (practical.js) real alignments from AFDB or upload
 *
 * Both talk to the same worker (gremlin-core.js) and the same parser (msa.js).
 * Everything matrix-shaped is drawn on a canvas: the original emitted one SVG
 * node per coupling, which is 1.8M nodes at L=64/A=21.
 */
(function (root) {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var fmtInt = function (n) {
    return (n === undefined || n === null || !isFinite(n)) ? '-' : Math.round(n).toLocaleString();
  };
  var fmtBytes = function (b) {
    return b < 1048576 ? (b / 1024).toFixed(0) + ' KB'
      : b < 1073741824 ? (b / 1048576).toFixed(0) + ' MB' : (b / 1073741824).toFixed(2) + ' GB';
  };

  /* Diverging ramp: red negative, white zero, blue positive. This is the
     original getColor(), which clamped to a fixed +-2 range. */
  function diverging(u) {
    if (u < -1) u = -1; else if (u > 1) u = 1;
    if (u < 0) { var g = Math.round(255 * (1 + u)); return [255, g, g]; }
    var h = Math.round(255 * (1 - u));
    return [h, h, 255];
  }
  function divCss(u) { var c = diverging(u); return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }

  /*
   * Sequential single-hue ramp, blank white -> blue -> near-black over [0, 1].
   * This is what a contact map wants and the diverging ramp is simply wrong for
   * it: after APC the score is a magnitude, not a signed quantity. Half the
   * off-diagonal cells come out slightly negative (measured: p50 = -0.04x the
   * anchor on the demo alignment) and those are correction artifacts, so they
   * are blank, not red.
   *
   * GAMMA > 1 is the part that matters. Only 6.8% of off-diagonal pairs reach a
   * quarter of the colour anchor, so the map should read as mostly empty with
   * the contacts standing out of it. The old 0.7 *lifted* the low end -- a
   * 0.25x noise cell painted at 0.38 intensity -- which flooded the background
   * with blue speckle that competed with the real arcs. At 1.6 the same cell
   * paints at 0.11 and the secondary-structure arcs are the first thing you see.
   *
   * Steps are one hue, light to dark, so the ordering survives greyscale
   * printing and colour-vision deficiency; there is no hue change to misread.
   */
  var SEQ_GAMMA = 1.6;
  var SEQ_LUT = (function () {
    var stops = [[255, 255, 255], [205, 226, 251], [158, 197, 244], [109, 167, 236],
                 [57, 135, 229], [37, 106, 191], [24, 79, 149], [13, 54, 107]];
    var lut = new Array(256), k, x, s, t, a, b;
    for (k = 0; k < 256; k++) {
      x = (k / 255) * (stops.length - 1);
      s = Math.min(stops.length - 2, Math.floor(x));
      t = x - s;
      a = stops[s]; b = stops[s + 1];
      lut[k] = [Math.round(a[0] + (b[0] - a[0]) * t),
                Math.round(a[1] + (b[1] - a[1]) * t),
                Math.round(a[2] + (b[2] - a[2]) * t)];
    }
    return lut;
  })();

  /* Returns a shared triple from the lookup table -- read it, do not mutate it. */
  function sequential(u) {
    if (!(u > 0)) return SEQ_LUT[0];
    if (u > 1) u = 1;
    return SEQ_LUT[Math.round(Math.pow(u, SEQ_GAMMA) * 255)];
  }
  function seqCss(u) { var c = sequential(u); return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }

  /** Size a canvas for the device pixel ratio and return a pre-scaled context. */
  function fitCanvas(cv, w, h) {
    var dpr = window.devicePixelRatio || 1;
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    cv.style.width = w + 'px';
    cv.style.height = h + 'px';
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  function innerW(node, fallback) {
    return Math.max(120, (node && node.clientWidth) || fallback || 320);
  }

  /* ------------------------------------------------------------------ */
  /* heat maps                                                          */
  /* ------------------------------------------------------------------ */

  /*
   * Reproduces the original <Heatmap>: cells, a black outline (per A x A block
   * for the coupling matrix), and numbered ticks on both axes.
   *
   * opts = {
   *   get(i, j)   -> value, over n x n cells
   *   n           number of cells per side
   *   size        pixel size of the plot area
   *   L, A        for block outlines and tick spacing
   *   blocks      draw the L x L block grid (the coupling matrix)
   *   scale       value mapped to full colour (the original used a fixed 2)
   *   gridStroke  draw the per-cell white separator (only legible when cells are big)
   *   ramp        'diverging' (default, for signed W) or 'sequential' (contact scores)
   * }
   */
  function drawHeatmap(cv, opts) {
    var size = opts.size, n = opts.n, L = opts.L, A = opts.A;
    var seq = opts.ramp === 'sequential';
    var rgb = seq ? sequential : diverging;
    var css = seq ? seqCss : divCss;
    var padL = 35, padT = 5, padB = 35, padR = 5;
    var ctx = fitCanvas(cv, size + padL + padR, size + padT + padB);
    var cell = size / n;
    var inv = opts.scale > 0 ? 1 / opts.scale : 0;
    var i, j, v, p;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, size + padL + padR, size + padT + padB);
    ctx.translate(padL, padT);

    /*
     * Big matrices are painted at native resolution into an offscreen canvas and
     * scaled up with smoothing off -- one fillRect per cell would be L^2 A^2
     * calls, which is 1.8M at L=64.
     */
    if (cell < 3) {
      var off = document.createElement('canvas');
      off.width = n; off.height = n;
      var img = new ImageData(n, n);
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          var c = rgb(opts.get(i, j) * inv), o = (i * n + j) * 4;
          img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
        }
      }
      off.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, n, n, 0, 0, size, size);
    } else {
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          ctx.fillStyle = css(opts.get(i, j) * inv);
          ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
        }
      }
      if (opts.gridStroke && cell >= 6) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        for (i = 0; i <= n; i++) {
          p = Math.round(i * cell) + 0.5;
          ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
        }
      }
    }

    // block outlines, or one outline round the whole plot
    var step = size / L;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    if (opts.blocks && step >= 4) {
      for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) ctx.strokeRect(j * step, i * step, step, step);
      }
    } else {
      ctx.strokeRect(0, 0, size, size);
    }

    // ticks, thinned so the labels never collide
    var every = Math.max(1, Math.ceil(L / Math.max(2, Math.floor(size / 26))));
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#000';
    ctx.font = '13px ui-monospace, monospace';
    for (i = 0; i < L; i += every) {
      p = i * step;
      ctx.beginPath(); ctx.moveTo(p, size); ctx.lineTo(p, size + 5); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(String(i), p + step / 2, size + 8);
      ctx.beginPath(); ctx.moveTo(-5, p); ctx.lineTo(0, p); ctx.stroke();
      ctx.textAlign = 'end'; ctx.textBaseline = 'middle';
      ctx.fillText(String(i), -8, p + step / 2);
    }
  }

  /* ------------------------------------------------------------------ */
  /* loss chart                                                         */
  /* ------------------------------------------------------------------ */

  /* Fed from the worker's decimating history buffer, so the point count is
     bounded no matter how many steps have run. */
  function drawLoss(cv, wrap, hist, opts) {
    opts = opts || {};
    var W = innerW(wrap, 420), H = opts.height || 200;
    var ctx = fitCanvas(cv, W, H);
    var padL = opts.padL || 70, padR = 12, padT = 10, padB = 32;
    var pw = Math.max(1, W - padL - padR), ph = Math.max(1, H - padT - padB);
    var k, v, x, y, g;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);

    var n = hist ? hist.length / 2 : 0;
    if (!n) {
      ctx.fillStyle = '#9ca3af';
      ctx.font = '13px ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('no steps yet', W / 2, H / 2);
      return;
    }

    var xmax = hist[(n - 1) * 2], ymin = Infinity, ymax = -Infinity;
    for (k = 0; k < n; k++) {
      v = hist[k * 2 + 1];
      if (isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
    }
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (ymax - ymin < 1e-9) ymax = ymin + 1e-9;
    var pad = (ymax - ymin) * 0.08;
    ymin -= pad; ymax += pad;
    if (xmax < 1) xmax = 1;

    var X = function (s) { return padL + (s / xmax) * pw; };
    var Y = function (t) { return padT + ph - ((t - ymin) / (ymax - ymin)) * ph; };

    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (g = 0; g <= 4; g++) {
      v = ymin + (ymax - ymin) * g / 4; y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), padL - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (g = 0; g <= 4; g++) {
      v = xmax * g / 4; x = Math.round(X(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke();
      ctx.fillText(fmtInt(v), x, padT + ph + 6);
    }

    ctx.strokeStyle = '#9ca3af';
    ctx.strokeRect(padL + 0.5, padT + 0.5, pw, ph);

    ctx.strokeStyle = '#8884d8';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (k = 0; k < n; k++) {
      x = X(hist[k * 2]); y = Y(hist[k * 2 + 1]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#374151';
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(opts.xLabel || 'Iterations', padL + pw / 2, H - 2);
    ctx.save();
    ctx.translate(12, padT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'top';
    ctx.fillText(opts.yLabel || '-PLL', 0, 0);
    ctx.restore();
  }

  /* ------------------------------------------------------------------ */
  /* dense MSA canvas (virtualized)                                     */
  /* ------------------------------------------------------------------ */

  /*
   * Draws only the rows currently scrolled into view, so cost is bounded by the
   * panel rather than by N*L. Optional coverage/identity bars on the left, in
   * the spirit of py2Dmol's MSA view.
   */
  function drawMsaRows(cv, box, ds, o) {
    o = o || {};
    var rowH = o.rowH || 9, gutter = o.gutter || 52, barW = o.bars ? 46 : 0;
    var H = o.height || 260;
    var W = innerW(box, 380);
    var ctx = fitCanvas(cv, W, H);
    var L = ds.L, N = ds.N, seqs = ds.seqs, colors = ds.colors;
    var left = gutter + barW;
    var cellW = Math.max(0.4, (W - left - 4) / L);
    var scroll = box.scrollTop;
    var first = Math.max(0, Math.floor(scroll / rowH));
    var last = Math.min(N, first + Math.ceil(H / rowH) + 1);
    var n, c, y, off;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);

    for (n = first; n < last; n++) {
      y = (n - first) * rowH - (scroll % rowH);
      off = n * L;
      for (c = 0; c < L; c++) {
        ctx.fillStyle = colors[seqs[off + c]] || '#eee';
        ctx.fillRect(left + c * cellW, y, Math.max(1, cellW), rowH - 0.5);
      }
      if (n === o.sel) {
        ctx.strokeStyle = '#111';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(left - 1, y - 0.5, W - left, rowH);
      }
    }

    // gutter + bars
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, left - 3, H);
    if (o.bars && ds.cov && ds.idn) {
      for (n = first; n < last; n++) {
        y = (n - first) * rowH - (scroll % rowH);
        ctx.fillStyle = '#93c5fd';
        ctx.fillRect(gutter, y, Math.max(1, (barW / 2 - 2) * ds.cov[n]), rowH - 1.5);
        ctx.fillStyle = '#fcd34d';
        ctx.fillRect(gutter + barW / 2, y, Math.max(1, (barW / 2 - 2) * ds.idn[n]), rowH - 1.5);
      }
    }
    ctx.font = '9px ui-monospace, monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'right';
    for (n = first; n < last; n++) {
      if (n % 5 !== 0 && n !== o.sel) continue;
      y = (n - first) * rowH - (scroll % rowH);
      ctx.fillStyle = n === o.sel ? '#111' : '#9ca3af';
      ctx.fillText(String(n), gutter - 8, y);
    }
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  }

  /* ------------------------------------------------------------------ */
  /* page switcher                                                      */
  /* ------------------------------------------------------------------ */

  function nav(active) {
    return '<div class="nav">'
      + '<a href="index.html"' + (active === 'edu' ? ' class="on"' : '') + '>Educational</a>'
      + '<a href="practical.html"' + (active === 'practical' ? ' class="on"' : '') + '>Practical</a>'
      + '<span class="nav-note">' + (active === 'edu'
        ? 'toy alignment, every intermediate shown'
        : 'real alignments from AlphaFold DB or your own file') + '</span>'
      + '</div>';
  }

  /* ------------------------------------------------------------------ */
  /* snapshots                                                          */
  /* ------------------------------------------------------------------ */

  /*
   * The worker only recomputes the expensive fields (coupling raster, top-K,
   * dense matrix) on every third snapshot, and cheap snapshots simply omit them.
   * Carry the last known values forward so those panels keep their content
   * instead of blanking between heavy frames. Dropped whenever the model is
   * reallocated, since a stale matrix would be the wrong size.
   */
  var HEAVY = ['coupImg', 'coupN', 'coupScale', 'top', 'wmat'];

  function mergeSnap(prev, next) {
    if (prev && prev.L === next.L && prev.A === next.A) {
      for (var i = 0; i < HEAVY.length; i++) {
        var k = HEAVY[i];
        if (next[k] === undefined && prev[k] !== undefined) next[k] = prev[k];
      }
    }
    return next;
  }

  /* ------------------------------------------------------------------ */
  /* ranked contacts                                                    */
  /* ------------------------------------------------------------------ */

  /** Sorted [i, j, score] triples with |i-j| >= minSep, capped at `limit`. */
  /*
   * `map` is optional: pass a column map and the sequence separation is measured
   * in the input alignment's numbering rather than in model columns. It matters
   * whenever a column filter has removed positions, because dropping columns
   * compresses the model -- two residues 6 apart in the protein can be 4 apart
   * in the model, and a "min |i-j| >= 5" cut would then throw away a genuinely
   * long-range pair. Without `map` the two are the same thing.
   */
  function rankContacts(contact, L, minSep, limit, map) {
    if (!contact) return [];
    var out = [], i, j;
    for (i = 0; i < L; i++) {
      for (j = i + 1; j < L; j++) {
        if ((map ? map[j] - map[i] : j - i) < minSep) continue;
        out.push([i, j, contact[i * L + j]]);
      }
    }
    out.sort(function (a, b) { return b[2] - a[2]; });
    return limit ? out.slice(0, Math.min(out.length, limit)) : out;
  }

  function contactCsv(rows, colMap) {
    var csv = 'rank,col_i,col_j,model_i,model_j,score\n';
    rows.forEach(function (r, k) {
      csv += (k + 1) + ',' + (colMap ? colMap[r[0]] : r[0]) + ',' + (colMap ? colMap[r[1]] : r[1])
        + ',' + r[0] + ',' + r[1] + ',' + r[2].toFixed(6) + '\n';
    });
    return csv;
  }

  function download(name, text, mime) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: mime || 'text/plain' }));
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  root.UI = {
    $: $, fmtInt: fmtInt, fmtBytes: fmtBytes,
    diverging: diverging, divCss: divCss, sequential: sequential, seqCss: seqCss,
    fitCanvas: fitCanvas, innerW: innerW,
    drawHeatmap: drawHeatmap, drawLoss: drawLoss, drawMsaRows: drawMsaRows,
    nav: nav, mergeSnap: mergeSnap,
    rankContacts: rankContacts, contactCsv: contactCsv, download: download
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
