/*
 * practical.js -- the practical page: real alignments, one output that matters.
 *
 * Input follows py2Dmol's conventions so the two agree on what an alignment is:
 *   - AlphaFold DB serves the a3m it used, at
 *     https://alphafold.ebi.ac.uk/files/msa/AF-<ACC>-F1-msa_v6.a3m (CORS '*')
 *   - a3m lowercase columns are insertions and are stripped on parse
 *   - coverage is the non-gap fraction of a row; identity is measured against
 *     the query; '-', '.', ' ' and 'X' all count as gaps
 *   - defaults: coverage >= 0.75, identity >= 0.15, sorted by identity
 *
 * Output is deliberately one thing: APC(L2norm(gauge-fixed w)), drawn large and
 * updated live while the optimizer runs, plus the ranked pairs behind it.
 */
(function () {
  'use strict';

  var $ = UI.$;

  var MEM_WARN = 6e8;
  var MEM_HARD = 1.6e9;
  var TBL_MS = 400;

  var S = {
    text: null,             // raw alignment text, never put in the DOM
    source: null,           // description of where it came from
    ds: null,
    snap: null,
    info: null,
    running: false,
    status: 'starting worker',
    err: null,
    sel: 0,
    minSep: 5,
    showTop: true,
    maxRate: 0,             // real alignments never hit a display cap; run flat out
    /*
     * batch 128 rather than 256: the Adam update is O(L^2 A^2) and independent
     * of batch, which is a ~100ms floor per step at L=155. Measured throughput
     * at that size is 252 seq/s at B=64, 330 at B=128, 361 at B=256 and 410 at
     * B=1024 -- so B=128 buys 91% of peak throughput while giving the contact
     * map three times as many updates to evolve through.
     */
    cfg: { alpha: 0.01, beta: 0.01, lr: 0.05, batch: 128, regMode: 'gremlin' },
    prep: { keepQueryColumns: true, maxColGap: 1, minCoverage: 0.75, minIdentity: 0.15,
            sortByIdentity: true, dedup: false }
  };

  var worker = null;
  var tblRows = [];
  var tblLast = 0;

  /* ---------------------------------------------------------------- */
  /* skeleton                                                         */
  /* ---------------------------------------------------------------- */

  document.getElementById('root').innerHTML = UI.nav('practical') + `
<div class="wrap">
  <div class="head">
    <h1>GREMLIN</h1>
    <div class="sub">pseudo-likelihood Potts model &middot; couplings &rarr; contacts</div>
  </div>

  <div class="banner err" id="err" hidden></div>

  <div class="card stats" id="stats"></div>
  <div class="banner warn" id="memWarn" hidden></div>
  <div class="banner info" id="meffWarn" hidden></div>

  <div class="card" style="margin-bottom:24px">
    <h2>Alignment</h2>
    <div class="note">Fetch the MSA that AlphaFold DB used for a prediction, or load your own FASTA / A3M.</div>
    <div class="row">
      <input type="text" id="acc" placeholder="UniProt accession, e.g. P0A7Y4" size="26">
      <button id="fetch">Fetch from AFDB</button>
      <label class="file-btn">Load file<input type="file" id="file" accept=".a3m,.fa,.fasta,.afa,.aln,.txt"></label>
      <button class="small" id="demo">Demo (RNase H)</button>
      <span class="small" id="src"></span>
    </div>

    <div class="filters">
      <div class="sl">
        <label>min coverage: <span class="val" id="vCov"></span></label>
        <input type="range" id="sCov" min="0" max="1" step="0.05">
      </div>
      <div class="sl">
        <label>min identity to query: <span class="val" id="vIdn"></span></label>
        <input type="range" id="sIdn" min="0" max="1" step="0.05">
      </div>
      <div class="sl">
        <label>max column gaps: <span class="val" id="vCol"></span></label>
        <input type="range" id="sCol" min="0.1" max="1" step="0.05">
      </div>
      <div class="col-checks">
        <label class="cb"><input type="checkbox" id="cQuery" checked> query columns only</label>
        <label class="cb"><input type="checkbox" id="cSort" checked> sort by identity</label>
        <label class="cb"><input type="checkbox" id="cDedup"> drop duplicates</label>
        <button class="dark" id="apply">Apply filters</button>
      </div>
    </div>

    <ul class="warnings" id="warns"></ul>

    <div id="msaBox" class="msa-view" style="height:230px;margin-top:12px">
      <canvas id="msaCv"></canvas><div id="msaSpacer"></div>
    </div>
    <div class="tiny" id="msaNote">
      <span class="key cov"></span> coverage &nbsp; <span class="key idn"></span> identity to query
    </div>
  </div>

  <div class="card" style="margin-bottom:24px">
    <h2>Optimization</h2>
    <div class="note" id="pace"></div>
    <div class="row">
      <button id="run" disabled>Start</button>
      <button id="reset" class="grey" disabled>Reset</button>
      <span class="status" id="status" style="margin:0"></span>
      <span class="badge" id="backend" title="Selected compute backend. Each candidate must reproduce the JS reference on a probe problem before it is used."></span>
      <label class="spacer small">max steps/s
        <select id="rate"></select>
      </label>
    </div>
    <div class="sliders" id="sliders"></div>
    <div class="live" id="live"></div>
  </div>

  <div class="grid-main">
    <div class="card">
      <h2>Contact map</h2>
      <div class="note mono">APC(L2norm(gauge-fixed <span class="w">w</span>))</div>
      <div class="row" style="margin-bottom:10px">
        <label class="cb"><input type="checkbox" id="showTop" checked> mark top L</label>
        <label class="small">min |i-j| <input type="number" id="minSep" min="1" max="30"></label>
      </div>
      <div id="cmWrap"><canvas id="cmCv" class="clickable"></canvas></div>
      <div class="tiny" id="cmNote"></div>
    </div>

    <div class="stack">
      <div class="card">
        <h2>Loss</h2>
        <div class="note"></div>
        <div id="lossWrap"><canvas id="lossCv"></canvas></div>
      </div>
      <div class="card">
        <h2>Ranked contacts</h2>
        <div class="row" style="justify-content:space-between;margin-bottom:8px">
          <span class="small" id="tblNote"></span>
          <button class="small" id="dl">download CSV</button>
        </div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>#</th><th>i</th><th>j</th><th class="r">score</th></tr></thead>
            <tbody id="tblBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>`;

  /* ---------------------------------------------------------------- */
  /* controls                                                         */
  /* ---------------------------------------------------------------- */

  var STATS = ['N', 'L', 'A', 'Meff', 'params', 'memory', 'steps', 'steps/s'];

  (function buildStats() {
    var box = $('stats');
    STATS.forEach(function (k) {
      var d = document.createElement('div');
      d.setAttribute('data-stat', k);
      d.innerHTML = '<div class="k">' + k + '</div><div class="v">-</div>';
      box.append(d);
    });
  })();

  var SLIDERS = [
    { key: 'alpha', label: '<span class="reg">&alpha;</span> (couplings)', log: true, min: 1e-4, max: 1, fmt: function (v) { return v.toExponential(1); } },
    { key: 'beta', label: '<span class="reg">&beta;</span> (bias)', log: true, min: 1e-4, max: 1, fmt: function (v) { return v.toExponential(1); } },
    { key: 'lr', label: 'learning rate', log: true, min: 1e-3, max: 0.5, fmt: function (v) { return v.toFixed(3); } },
    { key: 'batch', label: 'batch (sequences/step)', log: true, min: 8, max: 4096, int: true, fmt: UI.fmtInt }
  ];

  function sPos(sp, v) { return sp.log ? (Math.log(v) - Math.log(sp.min)) / (Math.log(sp.max) - Math.log(sp.min)) : v; }
  function sVal(sp, p) {
    var v = sp.log ? Math.exp(Math.log(sp.min) + p * (Math.log(sp.max) - Math.log(sp.min))) : p;
    return sp.int ? Math.round(v) : v;
  }

  (function buildSliders() {
    var box = $('sliders');
    SLIDERS.forEach(function (sp) {
      var wrap = document.createElement('div');
      wrap.className = 'sl';
      wrap.innerHTML = '<label>' + sp.label + ': <span class="val"></span></label>'
        + '<input type="range" min="' + (sp.log ? 0 : sp.min) + '" max="' + (sp.log ? 1 : sp.max)
        + '" step="' + (sp.log ? 0.001 : 0.01) + '">';
      var val = wrap.querySelector('.val'), rng = wrap.querySelector('input');
      val.textContent = sp.fmt(S.cfg[sp.key]);
      rng.value = sPos(sp, S.cfg[sp.key]);
      rng.addEventListener('input', function () {
        S.cfg[sp.key] = sVal(sp, Number(rng.value));
        val.textContent = sp.fmt(S.cfg[sp.key]);
        send({ type: 'config', cfg: S.cfg, maxRate: S.maxRate });
      });
      box.append(wrap);
    });
  })();

  (function buildRate() {
    var sel = $('rate');
    [0, 200, 50, 20, 10, 5, 1].forEach(function (r) {
      var o = document.createElement('option');
      o.value = r;
      o.textContent = r === 0 ? 'unlimited' : String(r);
      sel.append(o);
    });
    sel.value = String(S.maxRate);
    sel.addEventListener('change', function () {
      S.maxRate = Number(sel.value);
      send({ type: 'config', cfg: S.cfg, maxRate: S.maxRate });
    });
  })();

  /* filter controls */
  var FILT = [
    ['sCov', 'vCov', 'minCoverage', function (v) { return v <= 0 ? 'off' : v.toFixed(2); }],
    ['sIdn', 'vIdn', 'minIdentity', function (v) { return v <= 0 ? 'off' : v.toFixed(2); }],
    ['sCol', 'vCol', 'maxColGap', function (v) { return v >= 1 ? 'off' : v.toFixed(2); }]
  ];
  FILT.forEach(function (f) {
    var rng = $(f[0]), lab = $(f[1]);
    rng.value = S.prep[f[2]];
    lab.textContent = f[3](S.prep[f[2]]);
    rng.addEventListener('input', function () {
      S.prep[f[2]] = Number(rng.value);
      lab.textContent = f[3](S.prep[f[2]]);
    });
  });
  [['cQuery', 'keepQueryColumns'], ['cSort', 'sortByIdentity'], ['cDedup', 'dedup']].forEach(function (p) {
    var cb = $(p[0]);
    cb.checked = !!S.prep[p[1]];
    cb.addEventListener('change', function () { S.prep[p[1]] = cb.checked; });
  });

  /* ---------------------------------------------------------------- */
  /* MSA panel                                                        */
  /* ---------------------------------------------------------------- */

  var MSA_H = 230, ROW_H = 6;

  function drawMsa() {
    var cv = $('msaCv'), box = $('msaBox'), ds = S.ds;
    if (!ds) { UI.fitCanvas(cv, 1, 1); $('msaSpacer').style.height = '0px'; return; }
    $('msaSpacer').style.height = (ds.N * ROW_H) + 'px';
    $('msaSpacer').style.marginTop = (-MSA_H) + 'px';
    UI.drawMsaRows(cv, box, ds, { rowH: ROW_H, height: MSA_H, sel: S.sel, bars: true, gutter: 46 });
  }

  $('msaBox').addEventListener('scroll', drawMsa);
  $('msaBox').addEventListener('click', function (e) {
    if (!S.ds) return;
    var box = $('msaBox'), r = box.getBoundingClientRect();
    var n = Math.floor((e.clientY - r.top + box.scrollTop) / ROW_H);
    if (n >= 0 && n < S.ds.N) { S.sel = n; send({ type: 'select', sel: n }); drawMsa(); }
  });

  /* ---------------------------------------------------------------- */
  /* contact map -- the one output                                    */
  /* ---------------------------------------------------------------- */

  // plot geometry in CSS pixels, so the hover readout can invert it
  var cmGeom = null, cmNote = '';
  var CM_PAD = 36, CM_TOP = 3;

  function drawContacts() {
    var cv = $('cmCv'), snap = S.snap;
    if (!snap || !snap.contact) {
      UI.fitCanvas(cv, 1, 1);
      cmGeom = null;
      $('cmNote').textContent = '';
      return;
    }
    var L = snap.L, cm = snap.contact;
    var size = Math.max(280, Math.min(560, UI.innerW($('cmWrap'), 420) - 44));
    var pad = CM_PAD;
    var ctx = UI.fitCanvas(cv, size + pad + 6, size + pad + 6);

    /*
     * Anchor the colour scale at the L-th ranked score rather than at
     * max|value|. APC output is heavily tailed -- a converged run has a top
     * contact two or three orders of magnitude above the bulk -- so scaling by
     * the maximum leaves everything except a few cells white.
     */
    var vmax = 0, i, j, v;
    if (tblRows.length >= Math.min(L, 4)) vmax = Math.abs(tblRows[Math.min(L, tblRows.length) - 1][2]);
    if (!(vmax > 0)) {
      for (i = 0; i < L; i++) for (j = i + 1; j < L; j++) {
        v = Math.abs(cm[i * L + j]);
        if (v > vmax) vmax = v;
      }
    }
    var inv = vmax > 0 ? 1 / vmax : 0;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, size + pad + 6, size + pad + 6);
    ctx.translate(pad, CM_TOP);

    // native-resolution paint, scaled up with smoothing off
    var off = document.createElement('canvas');
    off.width = L; off.height = L;
    var img = new ImageData(L, L);
    for (var k = 0; k < L * L; k++) {
      var c = UI.sequential(cm[k] * inv);
      img.data[k * 4] = c[0]; img.data[k * 4 + 1] = c[1]; img.data[k * 4 + 2] = c[2]; img.data[k * 4 + 3] = 255;
    }
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, L, L, 0, 0, size, size);

    var cell = size / L, marked = 0, t, r;
    // Nothing is ranked in a freshly reset model: every score is 0, so "top L"
    // would just circle the first L pairs in index order.
    if (S.showTop && tblRows.length && tblRows[0][2] > 0) {
      ctx.strokeStyle = 'rgba(5,150,105,0.95)';
      ctx.lineWidth = Math.max(1, Math.min(2, cell / 3));
      marked = Math.min(L, tblRows.length);
      for (t = 0; t < marked; t++) {
        r = Math.max(1.6, cell * 0.45);
        ctx.beginPath(); ctx.arc((tblRows[t][1] + 0.5) * cell, (tblRows[t][0] + 0.5) * cell, r, 0, 6.2832); ctx.stroke();
        ctx.beginPath(); ctx.arc((tblRows[t][0] + 0.5) * cell, (tblRows[t][1] + 0.5) * cell, r, 0, 6.2832); ctx.stroke();
      }
    }
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1);

    // ticks in input-alignment numbering
    var map = S.ds && S.ds.colMap;
    var every = Math.max(1, Math.ceil(L / Math.max(2, Math.floor(size / 44))));
    ctx.fillStyle = '#374151';
    ctx.font = '11px ui-monospace, monospace';
    for (i = 0; i < L; i += every) {
      var lbl = String(map ? map[i] : i);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(lbl, (i + 0.5) * cell, size + 6);
      ctx.textAlign = 'end'; ctx.textBaseline = 'middle';
      ctx.fillText(lbl, -6, (i + 0.5) * cell);
    }

    cmGeom = { L: L, cell: cell };
    cmNote = L + ' × ' + L + ' columns'
      + (marked ? ' · top ' + marked + ' circled (|i-j| ≥ ' + S.minSep + ')' : '')
      + (vmax > 0 ? ' · colour saturates at ' + vmax.toFixed(2) : '')
      + ' · axes are input-alignment columns';
    $('cmNote').textContent = cmNote;
  }

  /* Hover readout. The map is the whole output here, so being able to read a
     cell beats a click target that does nothing. The canvas is sized in CSS
     pixels, so the plot area starts at (CM_PAD, CM_TOP) with no extra scaling. */
  $('cmCv').addEventListener('mousemove', function (e) {
    var snap = S.snap;
    if (!snap || !snap.contact || !cmGeom) return;
    var r = e.currentTarget.getBoundingClientRect();
    var i = Math.floor((e.clientY - r.top - CM_TOP) / cmGeom.cell);
    var j = Math.floor((e.clientX - r.left - CM_PAD) / cmGeom.cell);
    if (i < 0 || j < 0 || i >= cmGeom.L || j >= cmGeom.L) {
      $('cmNote').textContent = cmNote;
      return;
    }
    var map = S.ds && S.ds.colMap;
    $('cmNote').textContent = 'i ' + (map ? map[i] : i) + '   j ' + (map ? map[j] : j)
      + '   |i-j| ' + Math.abs(i - j)
      + '   score ' + snap.contact[i * cmGeom.L + j].toFixed(3);
  });
  $('cmCv').addEventListener('mouseleave', function () { $('cmNote').textContent = cmNote; });

  /* ---------------------------------------------------------------- */
  /* ranked contacts                                                  */
  /* ---------------------------------------------------------------- */

  function refreshTable(force) {
    var snap = S.snap;
    if (!snap || !snap.contact) { tblRows = []; $('tblBody').innerHTML = ''; $('tblNote').textContent = ''; return; }
    var now = Date.now();
    if (!force && now - tblLast < TBL_MS) return;
    tblLast = now;

    var all = UI.rankContacts(snap.contact, snap.L, S.minSep, 0);
    tblRows = all.slice(0, Math.min(all.length, 3 * snap.L));

    var map = S.ds && S.ds.colMap, body = $('tblBody');
    body.innerHTML = '';
    tblRows.slice(0, 20).forEach(function (r, k) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td class="dim">' + (k + 1) + '</td><td>' + (map ? map[r[0]] : r[0])
        + '</td><td>' + (map ? map[r[1]] : r[1]) + '</td><td class="r">' + r[2].toFixed(3) + '</td>';
      body.append(tr);
    });
    $('tblNote').textContent = 'top 20 of ' + all.length + ' pairs, |i-j| ≥ ' + S.minSep;
  }

  $('dl').addEventListener('click', function () {
    if (!tblRows.length) return;
    UI.download('contacts.csv', UI.contactCsv(tblRows, S.ds && S.ds.colMap), 'text/csv');
  });

  /* ---------------------------------------------------------------- */
  /* render                                                           */
  /* ---------------------------------------------------------------- */

  function setStat(k, v) {
    var n = document.querySelector('[data-stat="' + k + '"] .v');
    if (n) n.textContent = v;
  }

  /* Which backend the worker settled on, and why the others were not used. */
  function renderBackend() {
    var b = S.backend;
    var el = $('backend');
    if (!b) { el.textContent = ''; return; }
    el.textContent = b.name;
    el.className = 'badge ' + (b.name === 'js' ? 'badge-plain' : 'badge-fast');
    var why = (b.tried || []).map(function (t) {
      return t.name + ': ' + (t.ok ? 'ok (dev ' + t.dev.toExponential(1) + ')'
                                   : (t.err || 'dev ' + (t.dev || 0).toExponential(1) + ' too large'));
    });
    el.title = 'Compute backend: ' + b.name + '\n' + why.join('\n')
      + '\nA backend is only used if it reproduces the JS reference on a probe problem.';
  }

  function renderAll() {
    var ds = S.ds, snap = S.snap, info = S.info;

    $('err').hidden = !S.err;
    if (S.err) $('err').textContent = S.err;

    setStat('N', ds ? UI.fmtInt(ds.N) : '-');
    setStat('L', ds ? ds.L : '-');
    setStat('A', ds ? ds.A : '-');
    setStat('Meff', info ? UI.fmtInt(info.Meff) + (info.approxWeights ? '~' : '') : '-');
    setStat('params', info ? UI.fmtInt(info.params) : '-');
    var bytes = ds ? 16 * ds.L * ds.L * ds.A * ds.A : 0;
    setStat('memory', bytes ? UI.fmtBytes(bytes) : '-');
    setStat('steps', snap ? UI.fmtInt(snap.steps) : '-');
    setStat('steps/s', snap && snap.sps ? snap.sps.toFixed(1) : '-');

    $('memWarn').hidden = !(bytes > MEM_WARN);
    if (bytes > MEM_WARN) {
      $('memWarn').innerHTML = UI.fmtBytes(bytes) + ' of parameters. Full-rank Potts is O(L²A²), '
        + 'so this is the real ceiling on L in a browser tab — not the arithmetic.';
    }
    $('meffWarn').hidden = !(info && info.approxWeights);
    if (info && info.approxWeights) {
      $('meffWarn').innerHTML = 'M<sub>eff</sub> is approximate: exact reweighting is O(N²L), so above '
        + '3000 sequences neighbours are counted against a random reference subset and scaled.';
    }

    // Gate on `info`, not `ds`: reweighting is O(N^2 L) and takes ~10s on a
    // 15k-sequence alignment, and until it lands the worker has no model to run.
    var ready = !!(ds && info);
    $('run').disabled = !ready;
    $('reset').disabled = !ready;
    $('run').textContent = S.running ? 'Stop' : 'Start';
    $('run').className = S.running ? 'stop' : '';
    $('status').textContent = S.status;
    $('src').textContent = S.source || '';

    /*
     * Say how long this will take. At L=155 a step is about a second in scalar
     * JS, so a converged run is minutes, not seconds -- better to state that
     * than to leave someone watching a slow map and wondering if it is stuck.
     */
    if (snap && snap.sps > 0) {
      var spStep = 1 / snap.sps;
      var mins = 200 * spStep / 60;
      $('pace').textContent = spStep.toFixed(2) + ' s/step · 200 steps ≈ '
        + (mins < 1 ? (200 * spStep).toFixed(0) + ' s' : mins.toFixed(1) + ' min')
        + ' · watch the contact map settle rather than waiting for a fixed count';
    } else {
      $('pace').textContent = ds
        ? 'Cost per step is O(L²·A·batch) plus a fixed O(L²A²) parameter update.'
        : '';
    }

    $('live').textContent = snap
      ? '-PLL ' + snap.pll.toFixed(4) + '   L2(w) ' + snap.regW.toFixed(4)
        + '   L2(b) ' + snap.regB.toFixed(4) + '   total ' + snap.loss.toFixed(4)
        + '   rms(w) ' + snap.rms.toFixed(4)
      : '';

    var warns = $('warns');
    warns.innerHTML = '';
    if (ds) ds.warnings.forEach(function (w) {
      var li = document.createElement('li');
      li.textContent = w;
      warns.append(li);
    });

    drawMsa();
    refreshTable(false);
    drawContacts();
    UI.drawLoss($('lossCv'), $('lossWrap'), snap && snap.hist,
                { xLabel: 'steps', yLabel: 'loss', height: 170 });
  }

  /* ---------------------------------------------------------------- */
  /* worker                                                           */
  /* ---------------------------------------------------------------- */

  function send(m, xfer) { if (worker) worker.postMessage(m, xfer || []); }

  function startWorker() {
    try {
      worker = new Worker('gremlin-core.js');
    } catch (e) {
      S.err = 'Could not start the worker: ' + e.message
        + '. Serve this directory over HTTP (python3 -m http.server) rather than opening the file directly.';
      renderAll();
      return;
    }
    worker.onerror = function (e) {
      S.err = 'Worker error: ' + (e.message || 'unknown')
        + '. Serve this directory over HTTP rather than opening the file directly.';
      renderAll();
    };
    worker.onmessage = function (ev) {
      var d = ev.data;
      if (d.type === 'ready') {
        S.status = 'ready — load an alignment';
      } else if (d.type === 'progress') {
        S.status = d.phase === 'weights'
          ? 'computing sequence weights (' + (d.frac * 100).toFixed(0) + '%)'
          : d.phase;
        $('status').textContent = S.status;
        return;
      } else if (d.type === 'backend') {
        S.backend = d;
        renderBackend();
        return;
      } else if (d.type === 'inited') {
        S.info = d;
        S.status = 'ready';
      } else if (d.type === 'snapshot') {
        S.snap = UI.mergeSnap(S.snap, d);
        S.running = d.running;
      } else if (d.type === 'error') {
        S.err = d.message;
        S.running = false;
      } else {
        return;
      }
      renderAll();
    };
  }

  /* ---------------------------------------------------------------- */
  /* loading                                                          */
  /* ---------------------------------------------------------------- */

  function build() {
    if (S.text === null) return;
    S.err = null;
    var ds;
    try {
      ds = MSA.buildDataset(S.text, S.prep);
    } catch (e) {
      S.err = e.message;
      renderAll();
      return;
    }
    var bytes = 16 * ds.L * ds.L * ds.A * ds.A;
    if (bytes > MEM_HARD) {
      S.err = 'L=' + ds.L + ', A=' + ds.A + ' needs ' + UI.fmtBytes(bytes) + ' for W + gradient + '
        + 'Adam moments, which will not fit in a browser tab. Full-rank Potts is O(L²A²) in '
        + 'parameters, so the only lever is fewer columns: tighten "max column gaps", or trim the '
        + 'alignment before loading it.';
      renderAll();
      return;
    }

    S.ds = ds;
    S.sel = 0;
    S.snap = null;
    S.info = null;
    S.running = false;
    tblRows = [];
    tblLast = 0;
    if (ds.L - S.minSep < 1) { S.minSep = 1; $('minSep').value = 1; }
    S.status = 'initialising (' + UI.fmtInt(ds.N) + ' sequences)';
    renderAll();

    var copy = ds.seqs.slice();
    send({
      type: 'init', L: ds.L, A: ds.A, N: ds.N, seqs: copy,
      cfg: S.cfg, maxRate: S.maxRate, identity: 0.8, maxRefs: 3000, seed: 1234567,
      wantCoup: false, wantTop: false
    }, [copy.buffer]);
  }

  function fetchAfdb(acc) {
    acc = String(acc || '').trim().toUpperCase();
    if (!acc) { S.err = 'Enter a UniProt accession first.'; renderAll(); return; }
    S.err = null;
    S.status = 'fetching MSA for ' + acc + ' from AlphaFold DB…';
    renderAll();
    fetch(MSA.afdbMsaUrl(acc)).then(function (r) {
      if (r.status === 404) {
        throw new Error('AlphaFold DB has no MSA for ' + acc
          + '. Check the accession, or upload an alignment instead.');
      }
      if (!r.ok) throw new Error('AlphaFold DB returned HTTP ' + r.status + ' for ' + acc + '.');
      return r.text();
    }).then(function (text) {
      if (!text || !text.trim()) throw new Error('AlphaFold DB returned an empty MSA for ' + acc + '.');
      S.text = text;
      S.source = acc + ' · ' + UI.fmtInt((text.match(/^>/gm) || []).length) + ' sequences · '
        + UI.fmtBytes(text.length);
      build();
    }).catch(function (e) {
      S.err = e.message;
      S.status = 'ready';
      renderAll();
    });
  }

  $('fetch').addEventListener('click', function () { fetchAfdb($('acc').value); });
  $('acc').addEventListener('keydown', function (e) { if (e.key === 'Enter') fetchAfdb($('acc').value); });
  $('demo').addEventListener('click', function () {
    $('acc').value = 'P0A7Y4';
    fetchAfdb('P0A7Y4');
  });

  $('file').addEventListener('change', function (e) {
    var f = e.target.files && e.target.files[0];
    if (!f) return;
    var r = new FileReader();
    r.onload = function () {
      S.text = String(r.result);
      S.source = f.name + ' · ' + UI.fmtInt((S.text.match(/^>/gm) || []).length) + ' sequences · '
        + UI.fmtBytes(S.text.length);
      build();
    };
    r.readAsText(f);
  });

  $('apply').addEventListener('click', build);

  $('run').addEventListener('click', function () {
    if (S.running) { send({ type: 'pause' }); S.running = false; }
    else { send({ type: 'run' }); S.running = true; }
    renderAll();
  });
  $('reset').addEventListener('click', function () {
    send({ type: 'reset' });
    S.running = false;
    renderAll();
  });

  $('showTop').addEventListener('change', function (e) { S.showTop = e.target.checked; drawContacts(); });
  $('minSep').value = S.minSep;
  $('minSep').addEventListener('change', function (e) {
    S.minSep = Math.max(1, Math.min(30, Number(e.target.value) || 1));
    e.target.value = S.minSep;
    refreshTable(true);
    drawContacts();
  });

  var rz = null;
  window.addEventListener('resize', function () {
    clearTimeout(rz);
    rz = setTimeout(renderAll, 120);
  });

  renderAll();
  startWorker();
})();
