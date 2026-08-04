/*
 * edu.js -- the educational page.
 *
 * Layout, wording, slider ranges, defaults and step pacing are all kept as they
 * were in the original index.html. What changed is underneath:
 *
 *   - the optimizer runs in a worker (gremlin-core.js) instead of a 100ms
 *     setInterval on the UI thread
 *   - the weights and contact heat maps are canvases, not one <rect> per cell
 *   - the network diagram draws the top-K couplings, not all L(L-1)A^2 of them
 *   - softmax is max-subtracted, so large couplings no longer produce NaN
 *   - selecting a row that no longer exists can't throw
 *
 * To stay numerically comparable with the original it opts out of the two
 * modelling changes the practical page wants: regMode 'raw' (alpha is a bare
 * L2 coefficient, not scaled by (L-1)(A-1)/Meff) and uniformWeights (no
 * sequence reweighting, so Meff = N).
 */
(function () {
  'use strict';

  var $ = UI.$;

  var DEFAULT_MSA = '012\n102\n012\n102';
  var HEAT = 300;             // the original Heatmap size
  var HEAT_SCALE = 2;         // the original getColor() clamped to +-2
  var NET_MAX_LA = 320;
  var RATE = 10;              // the original ran on setInterval(step, 100)

  var S = {
    msa: DEFAULT_MSA,
    ds: null,
    snap: null,
    info: null,
    running: false,
    sel: 0,
    err: null,
    cfg: { alpha: 0.1, beta: 0.01, lr: 0.1, batch: 4096, regMode: 'raw' }
  };

  var worker = null;
  var debounce = null;

  /* ---------------------------------------------------------------- */
  /* skeleton -- same structure and copy as the original               */
  /* ---------------------------------------------------------------- */

  document.getElementById('root').innerHTML = UI.nav('edu') + `
<div class="wrap">
  <h1 class="edu-title">GREMLIN Visualization</h1>

  <div class="banner err" id="err" hidden></div>

  <!-- Row 1: Controls + Loss Chart -->
  <div class="grid2">
    <div class="stack">
      <div class="card">
        <h2>Optimization Controls</h2>
        <div class="row" style="margin-bottom:16px">
          <button id="run">Start Optimization</button>
          <button id="reset" class="grey">Reset</button>
          <span class="badge" id="backend"></span>
        </div>
        <div class="sliders3">
          <div class="sl">
            <label><span class="reg">&alpha;</span>: <span class="val" id="vAlpha"></span></label>
            <input type="range" id="sAlpha" min="0" max="0.5" step="0.01">
          </div>
          <div class="sl">
            <label><span class="reg">&beta;</span>: <span class="val" id="vBeta"></span></label>
            <input type="range" id="sBeta" min="0" max="0.5" step="0.01">
          </div>
          <div class="sl">
            <label>lr: <span class="val" id="vLr"></span></label>
            <input type="range" id="sLr" min="0.01" max="0.1" step="0.01">
          </div>
        </div>
      </div>
      <div class="card formula">
        x' = softmax(x@<span class="w">w</span>+<span class="b">b</span>)<br>
        loss = -xlog(x') + <span class="reg">&alpha;</span>L2(<span class="w">w</span>)
        + <span class="reg">&beta;</span>L2(<span class="b">b</span>)
      </div>
    </div>
    <div class="card">
      <h2>Loss Chart</h2>
      <div id="lossWrap"><canvas id="lossCv"></canvas></div>
    </div>
  </div>

  <!-- Row 2: MSA + Model -->
  <div class="grid2">
    <div class="card">
      <h2>Multiple Sequence Alignment</h2>
      <div class="msa-edit">
        <div class="msa-gutter" id="gutter"></div>
        <div class="msa-stage" id="stage">
          <canvas id="editCv"></canvas>
          <textarea id="editTa" spellcheck="false"></textarea>
        </div>
      </div>
    </div>
    <div class="card scrollx">
      <h2>Model</h2>
      <div id="modelPanel"></div>
    </div>
  </div>

  <!-- Row 3: Weights + Contact Map -->
  <div class="grid2">
    <div class="card">
      <h2>Weights</h2>
      <div class="heat-label"><span class="w">w</span></div>
      <canvas id="wCv"></canvas>
    </div>
    <div class="card">
      <h2>Contact Map</h2>
      <div class="heat-label">APC(L2norm(<span class="w">w</span>))</div>
      <canvas id="cmCv"></canvas>
    </div>
  </div>
</div>`;

  /* ---------------------------------------------------------------- */
  /* MSA editor: transparent textarea over a coloured canvas          */
  /* ---------------------------------------------------------------- */

  var LINE_H = 22, CHAR_W = 9, FONT = 15, PAD = 8;

  function drawEditor() {
    var cv = $('editCv'), stage = $('stage');
    var W = UI.innerW(stage, 380), H = 250;
    var ctx = UI.fitCanvas(cv, W, H);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);
    ctx.font = FONT + 'px ui-monospace, monospace';
    ctx.textBaseline = 'middle';

    var colors = MSA.DIGIT_COLORS;
    var lines = S.msa.split('\n');
    var rows = Math.min(lines.length, Math.ceil(H / LINE_H) + 1);
    var r, c, x, y, ln, k;

    for (r = 0; r < rows; r++) {
      y = PAD + r * LINE_H;
      ln = lines[r] || '';
      for (c = 0; c < ln.length; c++) {
        x = PAD + c * CHAR_W;
        if (x > W) break;
        k = ln.charCodeAt(c) - 48;
        if (k >= 0 && k <= 9 && colors[k]) {
          ctx.fillStyle = colors[k];
          ctx.fillRect(x, y, CHAR_W, LINE_H);
        }
        ctx.fillStyle = '#111';
        ctx.fillText(ln[c], x + 1, y + LINE_H / 2);
      }
    }
    ctx.strokeStyle = '#ccc';
    ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  }

  /*
   * One radio per line, as before -- but the selection is clamped to the number
   * of lines that actually exist. In the original, selecting row 4 and then
   * deleting it left selectedSequence pointing past the end of the filtered
   * array, and the next render threw on `.split` of undefined.
   */
  function buildGutter() {
    var g = $('gutter');
    var lines = S.msa.split('\n').length;
    var want = Math.min(lines, 200);
    if (S.sel >= want) S.sel = Math.max(0, want - 1);
    if (g.childElementCount === want) {
      var kids = g.children;
      for (var q = 0; q < kids.length; q++) kids[q].firstChild.checked = (S.sel === q);
      return;
    }
    g.innerHTML = '';
    for (var i = 0; i < want; i++) {
      var d = document.createElement('div');
      var r = document.createElement('input');
      r.type = 'radio';
      r.name = 'selectedSequence';
      r.checked = (S.sel === i);
      r.addEventListener('change', (function (idx) {
        return function () { S.sel = idx; send({ type: 'select', sel: idx }); };
      })(i));
      d.append(r);
      g.append(d);
    }
  }

  /* ---------------------------------------------------------------- */
  /* network diagram                                                  */
  /* ---------------------------------------------------------------- */

  var SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, text) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }

  function renderModel() {
    var snap = S.snap, ds = S.ds, panel = $('modelPanel');
    panel.innerHTML = '';
    if (!snap || !ds || !snap.probs || !snap.probs.length) return;
    if (snap.L * snap.A > NET_MAX_LA) {
      panel.innerHTML = '<div class="small">L&middot;A = ' + (snap.L * snap.A)
        + ' is too many nodes to draw; shorten the alignment or use the practical page.</div>';
      return;
    }

    var L = snap.L, A = snap.A;
    var xLoc = 100, xOff = 100, labelH = 30;
    var sp = Math.min(20, 300 / (L * A));
    var H = L * (sp * A + 20) + labelH;
    var yOf = function (i, a) { return i * (sp * A + 20) + a * sp + sp / 2; };
    var seqs = ds.seqs, sel = Math.min(snap.sel, ds.N - 1);
    var i, j, a, mid, state;

    var root = sv('svg', { width: 320, height: H, style: 'background:#fff' });
    root.append(
      sv('text', { x: xLoc, y: labelH / 1.5, 'font-size': 16, 'font-weight': 'bold', 'text-anchor': 'middle' }, 'x'),
      sv('text', { x: xLoc + xOff / 2, y: labelH / 1.5, 'font-size': 16, 'font-weight': 'bold', fill: 'blue', 'text-anchor': 'middle' }, 'w'),
      sv('text', { x: xLoc + xOff + sp * 0.5, y: labelH / 1.5, 'font-size': 16, 'font-weight': 'bold', fill: 'red', 'text-anchor': 'middle' }, 'b'),
      sv('text', { x: xLoc + xOff + sp * 2, y: labelH / 1.5, 'font-size': 16, 'font-weight': 'bold', 'text-anchor': 'middle' }, "x'")
    );
    var g = sv('g', { transform: 'translate(0,' + labelH + ')' });
    root.append(g);

    for (i = 0; i < L; i++) {
      state = seqs[sel * L + i];
      mid = i * (sp * A + 20) + (sp * A) / 2;
      for (a = 0; a < A; a++) {
        g.append(sv('line', {
          x1: 60, y1: mid, x2: xLoc, y2: yOf(i, a),
          stroke: a === state ? 'rgba(0,0,0,0.5)' : 'rgba(200,200,200,0.5)', 'stroke-width': 3
        }));
      }
      g.append(sv('rect', {
        x: 30, y: mid - 15, width: 30, height: 30,
        fill: ds.colors[state] || '#eee', stroke: 'black', 'stroke-width': 3
      }));
      // the original used text-anchor="center", which is not a legal SVG value
      g.append(sv('text', { x: 8, y: mid + 6, 'font-size': 16, 'text-anchor': 'start' }, String(i)));
    }

    // weakest first, so strong couplings draw on top
    var t = snap.top, lines = [], k, u;
    if (t) {
      for (k = 0; k < t.length; k += 5) lines.push([t[k], t[k + 1], t[k + 2], t[k + 3], t[k + 4]]);
      lines.sort(function (p, q) { return Math.abs(p[4]) - Math.abs(q[4]); });
      for (k = 0; k < lines.length; k++) {
        u = lines[k][4] / HEAT_SCALE;
        g.append(sv('line', {
          x1: xLoc + sp * 0.5, y1: yOf(lines[k][0], lines[k][1]),
          x2: xLoc + xOff - sp * 0.5, y2: yOf(lines[k][2], lines[k][3]),
          stroke: UI.divCss(u),
          'stroke-width': Math.min(Math.abs(lines[k][4]), 1) * 5,
          opacity: Math.min(Math.abs(lines[k][4]), 1)
        }));
      }
    }

    var grey = function (v) { return Math.round(255 * (1 - v)); };
    for (i = 0; i < L; i++) {
      state = seqs[sel * L + i];
      for (a = 0; a < A; a++) {
        g.append(sv('circle', {
          cx: xLoc, cy: yOf(i, a), r: sp / 2,
          fill: state === a ? '#000' : '#fff', stroke: '#000', 'stroke-width': 2
        }));
      }
      g.append(sv('rect', {
        x: xLoc + xOff, y: i * (sp * A + 20), width: sp * 2, height: sp * A,
        fill: 'none', stroke: 'black', 'stroke-width': 2
      }));
      for (a = 0; a < A; a++) {
        var pb = grey(snap.pBias[i * A + a]);
        var pn = grey(snap.pNoBias[i * A + a]);
        var pp = grey(snap.probs[i * A + a]);
        g.append(sv('circle', { cx: xLoc + xOff + sp * 0.5, cy: yOf(i, a), r: sp / 2, fill: 'rgb(255,' + pb + ',' + pb + ')', stroke: 'red', 'stroke-width': 2 }));
        g.append(sv('circle', { cx: xLoc + xOff, cy: yOf(i, a), r: sp / 2, fill: 'rgb(' + pn + ',' + pn + ',255)', stroke: 'blue', 'stroke-width': 2 }));
        g.append(sv('circle', { cx: xLoc + xOff + sp * 2, cy: yOf(i, a), r: sp / 2, fill: 'rgb(' + pp + ',' + pp + ',' + pp + ')', stroke: '#000', 'stroke-width': 2 }));
      }
    }
    panel.append(root);
  }

  /* ---------------------------------------------------------------- */
  /* heat maps                                                        */
  /* ---------------------------------------------------------------- */

  function drawWeights() {
    var snap = S.snap, cv = $('wCv');
    if (!snap || !snap.wmat) { UI.fitCanvas(cv, 1, 1); return; }
    var LA = snap.L * snap.A, m = snap.wmat;
    UI.drawHeatmap(cv, {
      n: LA, size: HEAT, L: snap.L, A: snap.A,
      blocks: true, gridStroke: true, scale: HEAT_SCALE,
      get: function (i, j) { return m[i * LA + j]; }
    });
  }

  function drawContacts() {
    var snap = S.snap, cv = $('cmCv');
    if (!snap || !snap.contact) { UI.fitCanvas(cv, 1, 1); return; }
    var L = snap.L, cm = snap.contact;
    /*
     * Sequential, not the signed red/blue ramp W uses: this panel is an APC'd
     * Frobenius norm, so it is a magnitude and its small negatives are
     * correction artifacts. Scaled to the largest off-diagonal score rather
     * than the fixed +-2 -- at toy sizes the scores are well under 2, so the
     * fixed scale left the whole map nearly blank.
     */
    var vmax = 0, i, j, v;
    for (i = 0; i < L; i++) {
      for (j = 0; j < L; j++) {
        if (i === j) continue;
        v = cm[i * L + j];
        if (v > vmax) vmax = v;
      }
    }
    UI.drawHeatmap(cv, {
      n: L, size: HEAT, L: L, A: snap.A,
      blocks: false, gridStroke: true, ramp: 'sequential',
      scale: vmax > 0 ? vmax : HEAT_SCALE,
      get: function (i, j) { return cm[i * L + j]; }
    });
  }

  /* ---------------------------------------------------------------- */
  /* render                                                           */
  /* ---------------------------------------------------------------- */

  function renderAll() {
    $('err').hidden = !S.err;
    if (S.err) $('err').textContent = S.err;
    $('run').disabled = !S.info;
    $('reset').disabled = !S.info;
    $('run').textContent = (S.running ? 'Stop' : 'Start') + ' Optimization';
    $('run').className = S.running ? 'stop' : '';
    $('vAlpha').textContent = S.cfg.alpha.toFixed(2);
    $('vBeta').textContent = S.cfg.beta.toFixed(2);
    $('vLr').textContent = S.cfg.lr.toFixed(3);
    drawEditor();
    buildGutter();
    UI.drawLoss($('lossCv'), $('lossWrap'), S.snap && S.snap.hist,
                { xLabel: 'Iterations', yLabel: '-PLL' });
    renderModel();
    drawWeights();
    drawContacts();
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
        rebuild(true);
      } else if (d.type === 'backend') {
        var el = $('backend');
        el.textContent = d.name;
        el.className = 'badge ' + (d.name === 'js' ? 'badge-plain' : 'badge-fast');
        el.title = 'Compute backend, chosen only after reproducing the JS reference on a probe problem.';
        return;
      } else if (d.type === 'inited') {
        S.info = d;
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

  /*
   * Edits apply immediately, as they did originally: a change that alters L or A
   * reallocates and resets, anything else swaps the sequences and keeps the
   * weights so you can watch the model react.
   */
  function rebuild(force) {
    S.err = null;
    var ds;
    try {
      ds = MSA.buildDataset(S.msa, { keepQueryColumns: false });
    } catch (e) {
      S.err = e.message;
      renderAll();
      return;
    }
    var dimsChanged = force || !S.ds || ds.L !== S.ds.L || ds.A !== S.ds.A;
    S.ds = ds;
    if (S.sel >= ds.N) S.sel = Math.max(0, ds.N - 1);

    var copy = ds.seqs.slice();
    if (dimsChanged) {
      S.snap = null;
      S.info = null;                 // re-gates Start until the new model exists
      S.running = false;
      send({
        type: 'init', L: ds.L, A: ds.A, N: ds.N, seqs: copy,
        cfg: S.cfg, maxRate: RATE, uniformWeights: true, seed: 1234567,
        wantCoup: false, wantTop: true,
        // digit mode has no gap state, and the original started the bias at zero
        gap: -1, biasInit: 'zero'
      }, [copy.buffer]);
    } else {
      send({ type: 'data', N: ds.N, seqs: copy, uniformWeights: true }, [copy.buffer]);
    }
    send({ type: 'select', sel: S.sel });
    renderAll();
  }

  /* ---------------------------------------------------------------- */
  /* wiring                                                           */
  /* ---------------------------------------------------------------- */

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

  [['sAlpha', 'alpha'], ['sBeta', 'beta'], ['sLr', 'lr']].forEach(function (p) {
    var el = $(p[0]);
    el.value = S.cfg[p[1]];
    el.addEventListener('input', function () {
      S.cfg[p[1]] = Number(el.value);
      send({ type: 'config', cfg: S.cfg, maxRate: RATE });
      renderAll();
    });
  });

  var ta = $('editTa');
  ta.value = S.msa;
  ta.addEventListener('input', function () {
    S.msa = ta.value;
    drawEditor();
    buildGutter();
    // debounced so a fast typist doesn't reallocate the model on every keypress
    clearTimeout(debounce);
    debounce = setTimeout(function () { rebuild(false); }, 250);
  });
  ta.addEventListener('scroll', function () {
    $('editCv').style.top = (-ta.scrollTop) + 'px';
    $('gutter').scrollTop = ta.scrollTop;
  });

  var rz = null;
  window.addEventListener('resize', function () {
    clearTimeout(rz);
    rz = setTimeout(renderAll, 120);
  });

  renderAll();
  startWorker();
})();
