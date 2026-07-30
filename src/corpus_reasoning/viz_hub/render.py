"""
Stage 3 — Renderer.

Reads ``outputs/data_examples.json`` + ``outputs/experiments.json`` and emits a
single self-contained ``outputs/index.html`` — all data embedded as JS consts,
inline CSS/JS, no server or external dependencies (open it straight in a browser).
Mirrors the EMO ``visualize.py`` "format": dark theme, header tab bar, sidebar +
detail panel.

Usage:
    python -m viz.render
"""

import json
import os

try:
    from . import config
except ImportError:
    import config


def safe_json(obj) -> str:
    """JSON-serialize, escaping </ so it can't break out of the <script> tag."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#0f1117; --surface:#1a1d27; --surface2:#232635; --border:#2e3347;
  --text:#e2e8f0; --dim:#8892a4; --accent:#4A90E2; --highlight:#fbbf24;
  --on:#27AE60; --on2:#E67E22; --on3:#E74C3C; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13px; height:100vh; display:flex; flex-direction:column; overflow:hidden; }
a { color:var(--accent); }

#header { background:var(--surface); border-bottom:1px solid var(--border); padding:10px 18px;
  display:flex; align-items:center; gap:16px; flex-shrink:0; }
#header h1 { font-size:15px; font-weight:600; white-space:nowrap; }
#header .meta { color:var(--dim); font-size:11px; }
.tabs { margin-left:auto; display:flex; gap:6px; }
.view-tab { background:var(--surface2); border:1px solid var(--border); color:var(--dim);
  padding:6px 16px; border-radius:6px; cursor:pointer; font-size:12px; user-select:none; }
.view-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }

#main { display:flex; flex:1; overflow:hidden; }
.view { display:none; flex:1; overflow:hidden; }
.view.active { display:flex; }

/* sidebar shared */
.sidebar { width:300px; flex-shrink:0; background:var(--surface); border-right:1px solid var(--border);
  display:flex; flex-direction:column; overflow-y:auto; }
.group-title { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
  color:var(--dim); padding:12px 14px 5px; }
.side-item { padding:8px 14px; cursor:pointer; border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:8px; }
.side-item:hover { background:var(--surface2); }
.side-item.selected { background:var(--surface2); border-left:3px solid var(--accent); padding-left:11px; }
.dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.side-label { font-size:12px; flex:1; }
.side-sub { color:var(--dim); font-size:10px; }

.content { flex:1; overflow-y:auto; padding:20px 26px; }
.detail-title { font-size:20px; font-weight:700; margin-bottom:4px; }
.detail-meta { color:var(--dim); font-size:12px; margin-bottom:14px; }
.desc { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:12px 16px; line-height:1.6; margin-bottom:16px; max-width:980px; }
.section-title { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--dim); margin:18px 0 8px; }

.badge { font-size:10px; padding:2px 8px; border-radius:10px; font-weight:600; color:#fff; }
.pill { display:inline-block; font-size:11px; padding:3px 9px; border-radius:12px;
  border:1px solid var(--border); background:var(--surface2); color:var(--dim); margin:2px 4px 2px 0; }

/* ladder chips */
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; }
.chip { background:var(--surface2); border:1px solid var(--border); color:var(--text);
  padding:5px 12px; border-radius:14px; cursor:pointer; font-size:12px; }
.chip.active { background:var(--accent); border-color:var(--accent); color:#fff; }
.chip .ct { color:var(--dim); font-size:10px; margin-left:5px; }
.chip.active .ct { color:#dbeafe; }

/* example cards */
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:14px 16px; margin-bottom:12px; }
.card-head { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
.kv { color:var(--dim); font-size:11px; }
.label-row { font-size:11px; font-weight:700; color:var(--dim); text-transform:uppercase;
  letter-spacing:.05em; margin:10px 0 4px; }
.prompt { font-family:'SF Mono','Fira Code',ui-monospace,monospace; font-size:12px; line-height:1.6;
  white-space:pre-wrap; word-break:break-word; background:var(--surface2); border:1px solid var(--border);
  border-radius:6px; padding:9px 12px; }
.doc { font-family:ui-monospace,monospace; font-size:11.5px; line-height:1.55; white-space:pre-wrap;
  word-break:break-word; border-left:2px solid var(--border); padding:3px 0 3px 10px; margin:5px 0; color:#cbd5e1; }
.doc .dt { color:var(--accent); font-weight:600; }
.answer { background:var(--highlight); color:#000; font-weight:700; padding:2px 8px; border-radius:4px;
  display:inline-block; margin:2px 4px 2px 0; font-family:ui-monospace,monospace; font-size:12px; }
.gold { font-family:ui-monospace,monospace; font-size:11px; color:var(--on); }

/* stats grid */
.stats { display:flex; gap:14px; flex-wrap:wrap; margin:14px 0 22px; }
.stat { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:16px 22px; min-width:150px; }
.stat .n { font-size:26px; font-weight:700; }
.stat .l { color:var(--dim); font-size:11px; margin-top:2px; }

/* tables */
table { border-collapse:collapse; font-size:12px; margin:6px 0 16px; }
th, td { border:1px solid var(--border); padding:6px 11px; text-align:center; }
th { background:var(--surface2); color:var(--dim); font-weight:600; }
td.rowlabel, th.rowlabel { text-align:left; white-space:nowrap; }
.cfgtable td.k { text-align:left; color:var(--dim); font-family:ui-monospace,monospace; }
.cfgtable td.v { text-align:left; font-family:ui-monospace,monospace; }

.ratiobar { height:14px; border-radius:7px; overflow:hidden; display:flex; max-width:420px;
  border:1px solid var(--border); margin:4px 0 2px; }
.ratiobar .sft { background:var(--accent); }
.ratiobar .cpt { background:var(--on2); }
.legend-row { display:flex; gap:16px; font-size:11px; color:var(--dim); margin-bottom:14px; }
.legend-row span b { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
.muted { color:var(--dim); }
.note { font-size:11px; color:var(--dim); margin-top:4px; }

/* ---------------- Results tab (searchable central table) ---------------- */
#view-results { flex-direction:column; }
#res-toolbar { display:flex; align-items:center; gap:14px; padding:9px 18px;
  background:var(--surface); border-bottom:1px solid var(--border); flex-shrink:0; }
#res-controls { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:flex-end;
  padding:10px 18px; background:var(--surface); border-bottom:1px solid var(--border); flex-shrink:0; }
#res-controls input[type=text], #res-controls select { background:var(--surface2);
  border:1px solid var(--border); color:var(--text); border-radius:6px; padding:6px 10px; font-size:12px; }
#res-search { min-width:260px; }
#res-controls .facet { display:flex; flex-direction:column; gap:2px; }
#res-controls .facet label { font-size:9px; text-transform:uppercase; letter-spacing:.05em; color:var(--dim); }
#res-facets { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:flex-end; }
#view-results button { background:var(--surface2); border:1px solid var(--border); color:var(--text);
  border-radius:6px; padding:6px 12px; font-size:12px; cursor:pointer; }
#view-results button:hover { border-color:var(--accent); }
#res-tablewrap { flex:1; overflow:auto; }
#res-table { border-collapse:collapse; width:100%; font-size:12px; margin:0; }
#res-table thead th { position:sticky; top:0; background:var(--surface2); border:none;
  border-bottom:2px solid var(--border); padding:8px 10px; text-align:left; white-space:nowrap;
  cursor:pointer; user-select:none; z-index:2; color:var(--dim); }
#res-table thead th:hover { color:var(--highlight); }
#res-table tbody td { padding:6px 10px; border:none; border-bottom:1px solid var(--border);
  text-align:left; vertical-align:top; max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#res-table tbody td.num { text-align:right; font-variant-numeric:tabular-nums; }
#res-table tbody tr { cursor:pointer; }
#res-table tbody tr:hover { background:var(--surface2); }
#res-table td.metricv { font-weight:700; color:var(--highlight); }
.restag { font-size:10px; padding:2px 7px; border-radius:10px; background:var(--surface2);
  border:1px solid var(--border); color:var(--dim); }
#res-colpicker { position:fixed; right:18px; top:132px; background:var(--surface); z-index:25;
  border:1px solid var(--border); border-radius:8px; padding:10px 14px; max-height:68vh; overflow:auto;
  display:none; box-shadow:0 8px 30px rgba(0,0,0,.5); }
#res-colpicker.open { display:block; }
#res-colpicker label { display:block; font-size:12px; padding:3px 0; cursor:pointer; }
#res-detail { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; z-index:30;
  align-items:center; justify-content:center; }
#res-detail.open { display:flex; }
#res-detailcard { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  width:min(720px,92vw); max-height:86vh; overflow:auto; padding:22px 26px; }
#res-detailcard table { width:100%; margin:0; }
#res-detailcard td { text-align:left; border:none; border-bottom:1px solid var(--border);
  white-space:normal; max-width:none; padding:6px 8px; }
#res-detailcard td.k { color:var(--dim); width:210px; vertical-align:top; font-family:ui-monospace,monospace; }
</style>
</head>
<body>

<div id="header">
  <h1>__TITLE__</h1>
  <span class="meta">__META__</span>
  <div class="tabs">
    <div class="view-tab active" data-view="results" onclick="switchTab('results')">Results</div>
    <div class="view-tab" data-view="overview" onclick="switchTab('overview')">Overview</div>
    <div class="view-tab" data-view="data" onclick="switchTab('data')">Data Explorer</div>
    <div class="view-tab" data-view="exp" onclick="switchTab('exp')">Experiments</div>
  </div>
</div>

<div id="main">
  <!-- OVERVIEW -->
  <div id="view-overview" class="view" style="overflow-y:auto">
    <div class="content" style="max-width:1000px">
      <div class="detail-title">Long-Context Corpus-Reasoning Suite</div>
      <div class="desc" id="ov-intro"></div>
      <div class="stats" id="ov-stats"></div>
      <div class="section-title">Task complexity (CTC) legend</div>
      <div class="legend-row" id="ov-legend"></div>
      <div class="section-title">Sections</div>
      <div class="desc">
        <b>Data Explorer</b> — every task in the suite, with real examples sampled across its
        context-length (or item-count) ladder. <br>
        <b>Experiments</b> — the CPT data-mixing runs that test whether mixing continued-pretraining
        text back into SFT recovers long-context (RULER) ability.
      </div>
    </div>
  </div>

  <!-- DATA EXPLORER -->
  <div id="view-data" class="view">
    <div class="sidebar" id="data-sidebar"></div>
    <div class="content" id="data-content"></div>
  </div>

  <!-- EXPERIMENTS -->
  <div id="view-exp" class="view">
    <div class="sidebar" id="exp-sidebar"></div>
    <div class="content" id="exp-content"></div>
  </div>

  <!-- RESULTS -->
  <div id="view-results" class="view active">
    <div id="res-toolbar">
      <span class="section-title" style="margin:0">Central results table</span>
      <span id="res-count" class="muted"></span>
      <span style="margin-left:auto"></span>
      <a href="results.csv" download>&#11015; results.csv</a>
      <button id="res-btn-clear" type="button">Clear filters</button>
      <button id="res-btn-cols" type="button">Columns &#9662;</button>
    </div>
    <div id="res-controls">
      <input type="text" id="res-search" placeholder="Full-text search across all columns&hellip;" autocomplete="off">
      <span id="res-facets"></span>
    </div>
    <div id="res-colpicker"></div>
    <div id="res-tablewrap">
      <table id="res-table"><thead><tr id="res-head"></tr></thead><tbody id="res-body"></tbody></table>
    </div>
    <div id="res-detail"><div id="res-detailcard">
      <h3 style="margin-bottom:10px">Result detail</h3>
      <table id="res-detailtbl"></table>
      <div style="margin-top:14px"><button id="res-btn-close" type="button">Close</button></div>
    </div></div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const EXPER = __EXPER_JSON__;
const RESULTS = __RESULTS_JSON__;
const ORDER_COLORS = {"O(N)":"var(--on)","O(N^2)":"var(--on2)","O(N^3+)":"var(--on3)"};

function esc(s){ return s==null ? '' : String(s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function fmtTok(n){ if(n==null) return '?'; if(n>=1e6) return (n/1e6).toFixed(n>=1e7?0:1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(n>=1e4?0:1)+'K'; return ''+n; }

function switchTab(name){
  document.querySelectorAll('.view-tab').forEach(t=>t.classList.toggle('active', t.dataset.view===name));
  document.getElementById('view-overview').classList.toggle('active', name==='overview');
  document.getElementById('view-data').classList.toggle('active', name==='data');
  document.getElementById('view-exp').classList.toggle('active', name==='exp');
  document.getElementById('view-results').classList.toggle('active', name==='results');
}

/* ---------------- Overview ---------------- */
function buildOverview(){
  const s = DATA.stats;
  document.getElementById('ov-intro').innerHTML =
    'A suite of long-context reasoning tasks spanning computational complexity classes, used to study how '
    + 'small LMs scale with corpus size and how modeling choices (chunked attention, CoT, data mixing) interact. '
    + 'Token estimates are &#8776; chars/4; synthetic-vocabulary tasks tokenize higher.';
  const cards = [
    [s.n_tasks, 'tasks'],
    [s.n_rungs, 'ladder rungs'],
    [fmtTok(s.max_ctx_tokens), 'max context (tok)'],
    [EXPER.experiments.length, 'CPT-mix experiments'],
  ];
  document.getElementById('ov-stats').innerHTML = cards.map(c=>
    `<div class="stat"><div class="n">${esc(c[0])}</div><div class="l">${esc(c[1])}</div></div>`).join('');
  const orders = [['O(N)','scales fine with long context'],
                  ['O(N^2)','all-pairs — shows the reasoning gap'],
                  ['O(N^3+)','combinatorial — intractable in the extreme']];
  document.getElementById('ov-legend').innerHTML = orders.map(o=>
    `<span><b style="background:${ORDER_COLORS[o[0]]}"></b>${esc(o[0])} — ${esc(o[1])}</span>`).join('');
}

/* ---------------- Data Explorer ---------------- */
let selTask=null, selRung=0;
function buildDataSidebar(){
  const sb = document.getElementById('data-sidebar');
  const byOrder = {};
  DATA.tasks.forEach(t=>{ (byOrder[t.order]=byOrder[t.order]||[]).push(t); });
  let html='';
  ['O(N)','O(N^2)','O(N^3+)'].forEach(ord=>{
    if(!byOrder[ord]) return;
    html += `<div class="group-title">${esc(ord)}</div>`;
    byOrder[ord].forEach(t=>{
      html += `<div class="side-item" data-key="${t.key}" onclick="selectTask('${t.key}')">`
        + `<div class="dot" style="background:${ORDER_COLORS[t.order]}"></div>`
        + `<span class="side-label">${esc(t.title)}</span>`
        + `<span class="side-sub">${t.rungs.length}</span></div>`;
    });
  });
  sb.innerHTML = html;
}
function selectTask(key){
  selTask=key; selRung=0;
  document.querySelectorAll('#data-sidebar .side-item').forEach(el=>
    el.classList.toggle('selected', el.dataset.key===key));
  renderTask();
}
function renderTask(){
  const t = DATA.tasks.find(x=>x.key===selTask);
  const c = document.getElementById('data-content');
  if(!t){ c.innerHTML='<div class="muted">Select a task.</div>'; return; }
  let html = `<div class="detail-title">${esc(t.title)}</div>`
    + `<div class="detail-meta"><span class="badge" style="background:${ORDER_COLORS[t.order]}">${esc(t.order)}</span>`
    + ` &nbsp; substrate: ${esc(t.substrate)}</div>`
    + `<div class="desc">${esc(t.description)}</div>`;
  html += `<div class="section-title">Context ladder &mdash; pick a length</div><div class="chips">`;
  t.rungs.forEach((r,i)=>{
    html += `<div class="chip ${i===selRung?'active':''}" onclick="selectRung(${i})">${esc(r.label)}`
      + `<span class="ct">&#8776;${fmtTok(r.approx_tokens)} tok</span></div>`;
  });
  html += `</div><div id="examples"></div>`;
  c.innerHTML = html;
  renderExamples();
}
function selectRung(i){ selRung=i; renderTask(); }
function renderExamples(){
  const t = DATA.tasks.find(x=>x.key===selTask);
  const r = t.rungs[selRung];
  const box = document.getElementById('examples');
  box.innerHTML = `<div class="section-title">${esc(r.label)} &middot; ${esc(r.file)} &middot; `
    + `${r.examples.length} example(s)</div>` + r.examples.map(renderExample).join('');
}
function renderExample(e){
  let h = `<div class="card"><div class="card-head">`
    + `<span class="badge" style="background:var(--accent)">${esc(e.source)}</span>`
    + `<span class="kv">${e.n_docs} docs</span>`
    + `<span class="kv">&#8776;${fmtTok(e.approx_tokens)} tokens</span></div>`;
  if(e.queries && e.queries.length){
    h += `<div class="label-row">Query / instruction</div>`;
    e.queries.forEach(q=>{ h += `<div class="prompt">${esc(q)}</div>`; });
  }
  if(e.documents && e.documents.length){
    h += `<div class="label-row">Documents (${e.n_docs_shown} of ${e.n_docs} shown)</div>`;
    e.documents.forEach((d,i)=>{
      h += `<div class="doc">`+(d.title?`<span class="dt">${esc(d.title)}</span><br>`:'')
        + `<span class="muted">[${i+1}]</span> ${esc(d.text)}</div>`;
    });
  }
  if(e.answers && e.answers.length){
    h += `<div class="label-row">Answer(s)</div>`;
    e.answers.forEach(a=>{ h += `<span class="answer">${esc(a)}</span>`; });
  }
  if(e.gold_doc_indices && (e.gold_doc_indices.length||0)){
    h += `<div class="label-row">Gold doc indices</div><div class="gold">${esc(JSON.stringify(e.gold_doc_indices))}</div>`;
  }
  const mk = Object.keys(e.meta||{});
  if(mk.length){
    h += `<div class="label-row">Metadata</div><div>`;
    mk.forEach(k=>{ h += `<span class="pill">${esc(k)}: ${esc(JSON.stringify(e.meta[k]))}</span>`; });
    h += `</div>`;
  }
  return h + `</div>`;
}

/* ---------------- Experiments ---------------- */
function scoreColor(v){
  if(v==null) return 'var(--surface)';
  const r = Math.round(231 + (39-231)*v), g = Math.round(76 + (174-76)*v), b = Math.round(60 + (96-60)*v);
  return `rgba(${r},${g},${b},0.35)`;
}
let selExp=null;
function buildExpSidebar(){
  const sb = document.getElementById('exp-sidebar');
  const byPhase = {};
  EXPER.experiments.forEach(e=>{ (byPhase[e.phase]=byPhase[e.phase]||[]).push(e); });
  let html='';
  Object.keys(byPhase).forEach(ph=>{
    html += `<div class="group-title">${esc(ph)}</div>`;
    byPhase[ph].forEach(e=>{
      const frac = (e.cpt_frac!=null)? Math.round(e.cpt_frac*100)+'% CPT' : '';
      html += `<div class="side-item" data-key="${e.key}" onclick="selectExp('${e.key}')">`
        + `<div class="dot" style="background:var(--on2)"></div>`
        + `<span class="side-label">${esc(e.title)}</span>`
        + `<span class="side-sub">${esc(frac)}</span></div>`;
    });
  });
  sb.innerHTML = html;
}
function rulerTable(){
  const rr = EXPER.ruler_results;
  if(!rr || !rr.rows) return '';
  let h = `<div class="section-title">${esc(rr.ruler_metric||'RULER')} by context length</div>`
    + `<table><tr><th class="rowlabel">variant</th>`
    + rr.rungs.map(c=>`<th>${esc(c)}</th>`).join('') + `</tr>`;
  rr.rows.forEach(row=>{
    h += `<tr><td class="rowlabel">${esc(row.label)}</td>`
      + row.scores.map(v=>`<td style="background:${scoreColor(v)}">${v==null?'<span class="muted">&mdash;</span>':v.toFixed(2)}</td>`).join('')
      + `</tr>`;
  });
  h += `</table><div class="note">null = pending (live numbers in wandb project memory-networks). Edit viz/results.json to fill in.</div>`;
  return h;
}
function selectExp(key){
  selExp=key;
  document.querySelectorAll('#exp-sidebar .side-item').forEach(el=>
    el.classList.toggle('selected', el.dataset.key===key));
  renderExp();
}
function renderExp(){
  const c = document.getElementById('exp-content');
  const e = EXPER.experiments.find(x=>x.key===selExp);
  let head = `<div class="desc">${esc(EXPER.narrative)}</div>` + rulerTable();
  if(!e){ c.innerHTML = head; return; }
  const cpt = (e.cpt_frac!=null)? e.cpt_frac : (e.config.CPT_FRAC?parseFloat(e.config.CPT_FRAC):0);
  const sftPct = Math.round((1-cpt)*100), cptPct = Math.round(cpt*100);
  let h = head + `<div class="detail-title" style="margin-top:18px">${esc(e.title)}</div>`
    + `<div class="detail-meta">${esc(e.script)} &middot; ${esc(e.commit)||'uncommitted'}</div>`
    + `<div class="desc">${esc(e.description)||'<span class="muted">(no description)</span>'}</div>`;
  h += `<div class="section-title">SFT / CPT token mix</div>`
    + `<div class="ratiobar"><div class="sft" style="width:${sftPct}%"></div><div class="cpt" style="width:${cptPct}%"></div></div>`
    + `<div class="legend-row"><span><b style="background:var(--accent)"></b>SFT ${sftPct}%</span>`
    + `<span><b style="background:var(--on2)"></b>CPT (dolma3longmino) ${cptPct}%</span></div>`;
  const ck = Object.keys(e.config||{});
  if(ck.length){
    h += `<div class="section-title">Config</div><table class="cfgtable">`;
    ck.forEach(k=>{ h += `<tr><td class="k">${esc(k)}</td><td class="v">${esc(e.config[k])}</td></tr>`; });
    h += `</table>`;
  }
  c.innerHTML = h;
}

/* ---------------- Results table ---------------- */
let resVisible = new Set(RESULTS.default_visible||[]);
let resSort = "date_eval_ran", resDir = -1;
const resFacetState = {};
const resByName = Object.fromEntries((RESULTS.cols||[]).map(c=>[c.name,c]));
const rEl = id => document.getElementById(id);

function resUnique(col){
  const s=new Set(); (RESULTS.rows||[]).forEach(r=>{const v=(r[col]||'').trim(); if(v) s.add(v);});
  const num = resByName[col] && resByName[col].numeric;
  return [...s].sort((a,b)=> num ? (parseFloat(a)-parseFloat(b)) : a.localeCompare(b));
}
function resBuildFacets(){
  const host=rEl('res-facets'); host.innerHTML='';
  (RESULTS.facets||[]).forEach(col=>{
    const wrap=document.createElement('span'); wrap.className='facet';
    const lab=document.createElement('label'); lab.textContent=col; wrap.appendChild(lab);
    const sel=document.createElement('select');
    sel.innerHTML='<option value="">all</option>'+resUnique(col).map(v=>`<option>${esc(v)}</option>`).join('');
    sel.onchange=()=>{resFacetState[col]=sel.value; resRender();};
    wrap.appendChild(sel); host.appendChild(wrap);
  });
}
function resBuildColPicker(){
  const host=rEl('res-colpicker'); host.innerHTML='<b>Show columns</b>';
  (RESULTS.cols||[]).forEach(c=>{
    const l=document.createElement('label');
    l.innerHTML=`<input type=checkbox ${resVisible.has(c.name)?'checked':''}> ${esc(c.name)}`;
    l.querySelector('input').onchange=e=>{e.target.checked?resVisible.add(c.name):resVisible.delete(c.name); resBuildHead(); resRender();};
    host.appendChild(l);
  });
}
function resBuildHead(){
  const tr=rEl('res-head'); tr.innerHTML='';
  (RESULTS.cols||[]).filter(c=>resVisible.has(c.name)).forEach(c=>{
    const th=document.createElement('th'); th.title=c.help||c.name;
    th.innerHTML=esc(c.name)+(resSort===c.name?` <span style="color:var(--accent)">${resDir>0?'▲':'▼'}</span>`:'');
    th.onclick=()=>{ if(resSort===c.name) resDir*=-1; else {resSort=c.name; resDir=1;} resBuildHead(); resRender(); };
    tr.appendChild(th);
  });
}
function resPass(r){
  for(const col in resFacetState){ if(resFacetState[col] && (r[col]||'').trim()!==resFacetState[col]) return false; }
  const q=rEl('res-search').value.trim().toLowerCase();
  if(q){ const hay=(RESULTS.cols||[]).map(c=>r[c.name]||'').join(' ').toLowerCase();
    if(!q.split(/\s+/).every(t=>hay.includes(t))) return false; }
  return true;
}
function resRender(){
  let rows=(RESULTS.rows||[]).filter(resPass);
  const num=resByName[resSort] && resByName[resSort].numeric;
  rows.sort((a,b)=>{ let x=a[resSort]||'', y=b[resSort]||'';
    if(num){ x=parseFloat(x); y=parseFloat(y); x=isNaN(x)?-Infinity:x; y=isNaN(y)?-Infinity:y; return (x-y)*resDir; }
    return x.localeCompare(y)*resDir; });
  const cols=(RESULTS.cols||[]).filter(c=>resVisible.has(c.name));
  rEl('res-body').innerHTML=rows.map(r=>'<tr data-i="'+RESULTS.rows.indexOf(r)+'">'+cols.map(c=>{
    let v=r[c.name]||''; let cls=c.numeric?'num':''; if(c.name==='metric_value') cls+=' metricv';
    let shown=v;
    if(c.numeric && v!=='' && /\.\d{5,}/.test(String(v))){ const f=parseFloat(v); if(!isNaN(f)) shown=String(Math.round(f*1e4)/1e4); }
    let disp = v==='' ? '<span class="muted">&mdash;</span>' : esc(shown);
    if((RESULTS.facets||[]).includes(c.name) && v) disp='<span class="restag">'+esc(shown)+'</span>';
    return '<td class="'+cls+'" title="'+esc(v)+'">'+disp+'</td>';
  }).join('')+'</tr>').join('');
  [...rEl('res-body').querySelectorAll('tr')].forEach(tr=>tr.onclick=()=>resDetail(RESULTS.rows[+tr.dataset.i]));
  rEl('res-count').textContent=rows.length+' / '+(RESULTS.rows||[]).length+' rows';
}
function resDetail(r){
  rEl('res-detailtbl').innerHTML=(RESULTS.cols||[]).map(c=>{
    let v=r[c.name]||'';
    if(c.name==='extra_metrics' && v){ try{ v='<pre style="white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:11px;color:var(--dim);margin:0">'+esc(JSON.stringify(JSON.parse(v),null,2))+'</pre>'; }catch(e){ v=esc(v);} }
    else v = v ? esc(v) : '<span class="muted">&mdash;</span>';
    return '<tr><td class="k" title="'+esc(c.help||'')+'">'+esc(c.name)+'</td><td>'+v+'</td></tr>';
  }).join('');
  rEl('res-detail').classList.add('open');
}
function buildResults(){
  if(!(RESULTS.rows||[]).length){
    rEl('res-tablewrap').innerHTML='<div style="padding:30px" class="muted">No results yet &mdash; push rows to the results-hub repo (results.csv).</div>';
    rEl('res-count').textContent='0 rows'; return;
  }
  resBuildFacets(); resBuildColPicker(); resBuildHead(); resRender();
  rEl('res-search').oninput=resRender;
  rEl('res-btn-cols').onclick=()=>rEl('res-colpicker').classList.toggle('open');
  rEl('res-btn-clear').onclick=()=>{ rEl('res-search').value=''; for(const k in resFacetState) delete resFacetState[k];
    document.querySelectorAll('#res-facets select').forEach(s=>s.value=''); resRender(); };
  rEl('res-btn-close').onclick=()=>rEl('res-detail').classList.remove('open');
  rEl('res-detail').onclick=e=>{ if(e.target.id==='res-detail') rEl('res-detail').classList.remove('open'); };
}

/* ---------------- init ---------------- */
buildOverview();
buildDataSidebar();
buildExpSidebar();
buildResults();
if(DATA.tasks.length) selectTask(DATA.tasks[0].key);
if(EXPER.experiments.length) selectExp(EXPER.experiments[0].key);
</script>
</body>
</html>"""


def render(data_examples: dict, experiments: dict, results: dict) -> str:
    stats = data_examples.get("stats", {})
    meta = (
        f"{stats.get('n_tasks', 0)} tasks &middot; "
        f"{stats.get('n_rungs', 0)} ladder rungs &middot; "
        f"{len(experiments.get('experiments', []))} CPT-mix experiments &middot; "
        f"{results.get('n', 0)} logged results"
    )
    html = TEMPLATE
    html = html.replace("__TITLE__", "Corpus-Reasoning · Long-Context Suite & CPT-Mixing")
    html = html.replace("__META__", meta)
    html = html.replace("__DATA_JSON__", safe_json(data_examples))
    html = html.replace("__EXPER_JSON__", safe_json(experiments))
    html = html.replace("__RESULTS_JSON__", safe_json(results))
    return html


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def main():
    config.ensure_out_dir()
    with open(config.DATA_EXAMPLES_JSON) as f:
        data_examples = json.load(f)
    with open(config.EXPERIMENTS_JSON) as f:
        experiments = json.load(f)
    results = _load_json(config.RESULTS_JSON,
                         {"rows": [], "cols": [], "facets": [], "default_visible": [], "n": 0})
    html = render(data_examples, experiments, results)
    with open(config.SITE_HTML, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(config.SITE_HTML) / 1e6
    print(f"[render] wrote {config.SITE_HTML} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
