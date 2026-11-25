/*
 * mmseqs.js -- build an alignment with the ColabFold MMseqs2 server.
 *
 * The same API ColabFold and localfold use (../localfold,
 * shared/input/mmseqs2-api.js, which this follows): POST the query to
 * ticket/msa, poll ticket/<id> until COMPLETE, then download
 * result/download/<id>, a gzipped tar holding one a3m per database searched.
 * CORS is open, so the browser talks to it directly.
 *
 * The public server is shared and rate-limited. It is meant for non-commercial
 * use and should not be hammered: one search per click, polled every 5-10s.
 *
 * Loads in node as well (module.exports) so the tar and a3m handling is testable
 * without a network.
 */
(function (root) {
  'use strict';

  var DEFAULT_API = 'https://api.colabfold.com/';
  var QUERY_ID = 101;
  var TRANSIENT_SUBMIT = { UNKNOWN: 1, RATELIMIT: 1 };
  var TRANSIENT_JOB = { UNKNOWN: 1, PENDING: 1, RUNNING: 1, RATELIMIT: 1 };
  var ATTEMPTS = 5;

  /* Accept a bare sequence or FASTA; return the uppercase residues or throw. */
  function normalize(seq) {
    var s = String(seq || '').replace(/^>.*$/gm, '').replace(/[\s\d*]+/g, '').toUpperCase();
    if (!s) throw new Error('Enter a protein sequence.');
    if (!/^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$/.test(s)) {
      throw new Error('That does not look like a protein sequence (only amino-acid letters are allowed).');
    }
    return s;
  }

  /* Heuristic: a UniProt accession is short and has a digit; a sequence is
     long and all letters. */
  function looksLikeSequence(text) {
    var s = String(text || '').trim();
    if (s.charAt(0) === '>') return true;
    s = s.replace(/\s+/g, '');
    return s.length >= 20 && /^[A-Za-z*]+$/.test(s);
  }

  function sleep(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal && signal.aborted) return reject(abortError());
      var t = setTimeout(function () {
        if (signal) signal.removeEventListener('abort', onAbort);
        resolve();
      }, ms);
      function onAbort() { clearTimeout(t); reject(abortError()); }
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
    });
  }

  function abortError() {
    var e = new Error('MMseqs2 search cancelled');
    e.name = 'AbortError';
    return e;
  }

  /* fetch with retries on network failure and 429/5xx, exponential backoff. */
  function request(url, init, label) {
    var last;
    function attempt(k) {
      return fetch(url, init).then(function (r) {
        if (r.ok) return r;
        last = new Error(label + ' failed: HTTP ' + r.status);
        if ([429, 500, 502, 503, 504].indexOf(r.status) < 0) throw last;
        return retry(k);
      }, function (e) {
        if (init && init.signal && init.signal.aborted) throw abortError();
        last = new Error(label + ' failed: could not reach ' + new URL(url).host + '. Check the connection.');
        return retry(k);
      });
    }
    function retry(k) {
      if (k + 1 >= ATTEMPTS) throw last;
      return sleep(500 * Math.pow(2, k), init && init.signal).then(function () { return attempt(k + 1); });
    }
    return attempt(0);
  }

  function json(r, label) {
    return r.json().catch(function () {
      throw new Error(label + ': the answer could not be read (an outage or sign-in page may have replied).');
    });
  }

  function ticketOf(v) {
    return {
      status: v && typeof v.status === 'string' ? v.status.toUpperCase() : 'ERROR',
      id: v && typeof v.id === 'string' && v.id ? v.id : undefined
    };
  }

  function gunzip(buf) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser cannot decompress the MMseqs2 result (needs DecompressionStream).');
    }
    var stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).arrayBuffer().then(function (b) { return new Uint8Array(b); });
  }

  /* Regular files of a POSIX tar, as { path: Uint8Array }. */
  function readTar(bytes) {
    var dec = new TextDecoder(), files = {}, off = 0;
    function str(a, n) {
      var end = bytes.indexOf(0, a);
      return dec.decode(bytes.subarray(a, end < 0 || end > a + n ? a + n : end));
    }
    while (off + 512 <= bytes.length) {
      var name = str(off, 100);
      if (!name) break;                                    // end-of-archive blocks
      var prefix = str(off + 345, 155);
      var size = parseInt(str(off + 124, 12).trim() || '0', 8);
      var type = bytes[off + 156];
      var start = off + 512;
      if (!(size >= 0) || start + size > bytes.length) throw new Error('MMseqs2 result is a truncated archive');
      if (type === 0 || type === 48) {
        files[(prefix ? prefix + '/' : '') + name.replace(/^\.\//, '')] = bytes.subarray(start, start + size);
      }
      off = start + Math.ceil(size / 512) * 512;
    }
    return files;
  }

  /* One query's block from a NUL-separated multi-query a3m. */
  function queryBlock(text, id) {
    var blocks = text.replace(/\r/g, '').split(/\0+/);
    var re = new RegExp('^>' + id + '(\\s|$)');
    for (var k = 0; k < blocks.length; k++) {
      var b = blocks[k].trim();
      if (re.test(b)) return b;
    }
    throw new Error('MMseqs2 result does not contain the query');
  }

  /*
   * The archive's a3ms as one alignment: UniRef first, then the environmental
   * databases (BFD/MGnify/MetaEuk/SMAG) with their repeated query record
   * dropped, so the query appears exactly once, as the first row.
   */
  function a3mFromTar(tarBytes) {
    var files = readTar(tarBytes), dec = new TextDecoder();
    function find(suffix) {
      for (var p in files) if (p === suffix || p.slice(-suffix.length - 1) === '/' + suffix) return files[p];
      return null;
    }
    var uniref = find('uniref.a3m');
    if (!uniref) throw new Error('MMseqs2 result is missing uniref.a3m');
    var out = queryBlock(dec.decode(uniref), QUERY_ID);
    var env = find('bfd.mgnify30.metaeuk30.smag30.a3m');
    if (env) {
      var e = queryBlock(dec.decode(env), QUERY_ID);
      var second = e.indexOf('\n>');                      // skip the query record
      if (second >= 0) out += '\n' + e.slice(second + 1);
    }
    return out + '\n';
  }

  /*
   * Search one sequence. onProgress({ phase, status, elapsed }) is called as it
   * goes; phases are submitting, queued, running, downloading, complete.
   * Resolves to { a3m, ticket, seconds }.
   */
  function search(sequence, opts) {
    opts = opts || {};
    var seq = normalize(sequence);
    var api = new URL(opts.apiUrl || DEFAULT_API);
    var signal = opts.signal;
    var t0 = Date.now(), ticket, status;
    function report(phase) {
      if (opts.onProgress) opts.onProgress({ phase: phase, status: status, elapsed: (Date.now() - t0) / 1000 });
    }
    function jitter() { return 5000 + Math.floor(Math.random() * 5000); }

    function submit() {
      status = 'SUBMIT'; report('submitting');
      return request(new URL('ticket/msa', api), {
        method: 'POST', signal: signal,
        headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
        body: new URLSearchParams({ q: '>' + QUERY_ID + '\n' + seq + '\n', mode: opts.env === false ? 'all' : 'env' })
      }, 'MMseqs2 submission').then(function (r) { return json(r, 'MMseqs2 submission'); }).then(function (v) {
        var t = ticketOf(v);
        status = t.status;
        if (TRANSIENT_SUBMIT[status]) { report('retrying'); return sleep(jitter(), signal).then(submit); }
        if (status === 'ERROR') throw new Error('MMseqs2 rejected the sequence or is temporarily unavailable.');
        if (status === 'MAINTENANCE') throw new Error('The MMseqs2 server is undergoing maintenance; try again later.');
        if (!t.id) throw new Error('MMseqs2 returned ' + status + ' without a ticket.');
        ticket = t.id;
      });
    }
    function poll() {
      if (!TRANSIENT_JOB[status]) return Promise.resolve();
      report(status === 'RUNNING' ? 'running' : 'queued');
      return sleep(jitter(), signal).then(function () {
        return request(new URL('ticket/' + encodeURIComponent(ticket), api), { signal: signal }, 'MMseqs2 status');
      }).then(function (r) { return json(r, 'MMseqs2 status'); }).then(function (v) {
        status = ticketOf(v).status;
        return poll();
      });
    }
    return submit().then(poll).then(function () {
      if (status !== 'COMPLETE') throw new Error('MMseqs2 search ended with status ' + status + '.');
      report('downloading');
      return request(new URL('result/download/' + encodeURIComponent(ticket), api), { signal: signal },
                     'MMseqs2 download');
    }).then(function (r) { return r.arrayBuffer(); }).then(gunzip).then(function (tar) {
      var a3m = a3mFromTar(tar);
      report('complete');
      return { a3m: a3m, ticket: ticket, seconds: (Date.now() - t0) / 1000 };
    });
  }

  /* The canonical sequence for a UniProt accession, from UniProt's REST API. */
  function uniprotSequence(acc, signal) {
    acc = String(acc || '').trim().toUpperCase();
    return request('https://rest.uniprot.org/uniprotkb/' + encodeURIComponent(acc) + '.fasta',
                   { signal: signal }, 'UniProt lookup').then(function (r) { return r.text(); }).then(function (t) {
      var s = t.replace(/^>.*$/gm, '').replace(/\s+/g, '');
      if (!s) throw new Error('UniProt has no sequence for ' + acc + '.');
      return s;
    });
  }

  root.MMseqs = {
    search: search,
    uniprotSequence: uniprotSequence,
    normalize: normalize,
    looksLikeSequence: looksLikeSequence,
    readTar: readTar,
    a3mFromTar: a3mFromTar,
    API: DEFAULT_API
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.MMseqs;
})(typeof globalThis !== 'undefined' ? globalThis : this);
