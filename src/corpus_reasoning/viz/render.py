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

/* hypothesis sections */
.hsec { background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:8px; padding:14px 18px; margin:0 0 18px; max-width:1180px; }
.hsec-title { font-size:16px; font-weight:700; margin-bottom:8px; }
.hyp { font-size:12.5px; line-height:1.55; margin-bottom:10px; }
.hyp b, .take b { color:var(--accent); }
.take { font-size:12.5px; line-height:1.55; margin-top:8px; padding-top:8px;
  border-top:1px solid var(--border); color:#cbd5e1; }
td.degraded { outline:2px solid var(--on3); outline-offset:-2px; }
.flag { color:var(--on3); font-weight:700; margin-left:3px; }
.refbar { font-size:11px; color:var(--dim); margin:2px 0 16px; max-width:1180px; }
.refbar b { color:var(--on); }

/* experiments sub-tabs */
.subtabs { display:flex; gap:6px; padding:10px 26px 0; flex-shrink:0; border-bottom:1px solid var(--border); }
.subtab { background:transparent; border:1px solid var(--border); border-bottom:none; color:var(--dim);
  padding:7px 16px; border-radius:6px 6px 0 0; cursor:pointer; font-size:12px; user-select:none; }
.subtab.active { background:var(--surface2); color:var(--text); border-color:var(--accent); }

/* doc-chunked attention viz */
.dc-toggle { display:flex; gap:6px; margin:6px 0 16px; }
.dc-toggle .opt { background:var(--surface2); border:1px solid var(--border); color:var(--dim);
  padding:6px 16px; border-radius:6px; cursor:pointer; font-size:12px; }
.dc-toggle .opt.active { background:var(--accent); color:#fff; border-color:var(--accent); }
.tokstrip { display:flex; flex-wrap:wrap; gap:4px; margin:6px 0 14px; max-width:1240px; }
.tok { font-family:ui-monospace,monospace; font-size:11px; padding:3px 7px; border-radius:5px;
  border:1px solid var(--border); white-space:nowrap; }
.tok .ix { color:var(--dim); font-size:9px; margin-right:4px; }
.tok.free { background:#1e293b; border-color:#334155; color:#cbd5e1; }
.tok.marker { background:transparent; border-color:var(--accent); color:var(--accent); font-weight:600; }
.tok.landmark { background:var(--highlight); color:#000; border-color:#b45309; font-weight:700; }
.tok.pad { background:var(--surface); border-style:dashed; color:var(--dim); }
.tok.ctx { color:#fff; font-weight:600; }
.dclegend { display:flex; flex-wrap:wrap; gap:14px; font-size:11px; color:var(--dim); margin:2px 0 14px; }
.dclegend span b { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px;
  vertical-align:-1px; }
/* mask grid */
.maskwrap { overflow:auto; max-width:100%; padding-bottom:8px; }
table.mask { border-collapse:collapse; margin:4px 0; }
table.mask td, table.mask th { border:1px solid #1c2030; padding:0; text-align:center; }
table.mask th { background:transparent; color:var(--dim); font-size:9px; font-weight:500;
  font-family:ui-monospace,monospace; height:auto; padding:1px 3px; border:none; white-space:nowrap; }
table.mask th.colh { writing-mode:vertical-rl; transform:rotate(180deg); max-height:120px; padding:3px 1px; }
table.mask td.cell { width:15px; height:15px; }
table.mask td.cell.on { background:var(--on); }
table.mask td.cell.off { background:#161922; }
table.mask td.cell.diag { outline:1px solid var(--highlight); outline-offset:-1px; }
table.mask th.memcol { color:var(--highlight); font-weight:700; }
table.mask tr.memrow th.rowh { color:var(--highlight); font-weight:700; }
.dc-cap { font-size:12px; color:#cbd5e1; line-height:1.55; max-width:1100px; margin:4px 0 8px; }

/* headline 32k matrix */
.hl-rungsel { margin:6px 0 14px; font-size:12px; color:var(--dim); }
.hl-rungsel .rlbl { margin-right:8px; font-weight:600; }
.hl-rungsel button.rungbtn { background:var(--surface2); color:var(--text); border:1px solid var(--border);
  border-radius:5px; padding:3px 11px; margin-right:5px; cursor:pointer; font-size:12px; font-weight:600; }
.hl-rungsel button.rungbtn:hover { border-color:var(--highlight); }
.hl-rungsel button.rungbtn.active { background:var(--highlight); color:#fff; border-color:var(--highlight); }
.hl-title { font-size:24px; font-weight:800; margin-bottom:4px; letter-spacing:-.01em; }
.hl-sub { color:var(--dim); font-size:13px; margin-bottom:14px; }
table.hl { border-collapse:collapse; font-size:12.5px; margin:8px 0 14px; min-width:880px; }
table.hl th, table.hl td { border:1px solid var(--border); padding:7px 12px; text-align:center; }
table.hl th { background:var(--surface2); color:var(--dim); font-weight:600; }
table.hl td.rl, table.hl th.rl { text-align:left; white-space:nowrap; }
table.hl tr.grp td { background:var(--surface2); text-align:left; font-weight:700; color:var(--text);
  text-transform:uppercase; letter-spacing:.05em; font-size:10.5px; padding:6px 12px; }
table.hl tr.modelrow:hover td { filter:brightness(1.06); }
table.hl tr.modelrow.expandable { cursor:pointer; }
.hl-rawname { color:var(--dim); font-size:10px; font-family:ui-monospace,monospace; display:block;
  margin-top:2px; }
.hl-vdot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px;
  vertical-align:0px; }
.hl-st { font-size:9.5px; padding:1px 7px; border-radius:9px; font-weight:700; margin-left:7px;
  vertical-align:1px; }
.hl-st.running { background:var(--on2); color:#000; }
.hl-st.partial { background:var(--highlight); color:#000; }
.hl-st.base { background:var(--surface2); color:var(--dim); border:1px solid var(--border); }
.hl-st.suspect { background:var(--on3); color:#fff; }
table.hl td.hl-na { color:var(--on3); font-weight:700; background:repeating-linear-gradient(
  45deg, transparent, transparent 4px, rgba(231,76,60,0.10) 4px, rgba(231,76,60,0.10) 8px); }
.hl-exp { color:var(--dim); font-size:10px; margin-left:5px; }
.hl-note { font-size:10.5px; color:var(--dim); }
.hl-ladder { background:var(--bg); }
.hl-ladder table.lad { border-collapse:collapse; font-size:11px; margin:6px 0 8px 22px; }
.hl-ladder table.lad th, .hl-ladder table.lad td { border:1px solid var(--border); padding:3px 9px;
  text-align:center; }
.hl-ladder table.lad th { background:var(--surface); color:var(--dim); }
.hl-ladder table.lad td.rl { text-align:left; white-space:nowrap; color:var(--text); }
.hl-foot { font-size:11px; color:var(--dim); line-height:1.6; max-width:1180px; margin:6px 0 18px; }
.hl-foot li { margin:2px 0; }
</style>
</head>
<body>

<div id="header">
  <h1>__TITLE__</h1>
  <span class="meta">__META__</span>
  <div class="tabs">
    <div class="view-tab active" data-view="headline" onclick="switchTab('headline')">Headline · 32k matrix</div>
    <div class="view-tab" data-view="overview" onclick="switchTab('overview')">Overview</div>
    <div class="view-tab" data-view="data" onclick="switchTab('data')">Data Explorer</div>
    <div class="view-tab" data-view="exp" onclick="switchTab('exp')">Experiments</div>
  </div>
</div>

<div id="main">
  <!-- HEADLINE · 32k matrix -->
  <div id="view-headline" class="view active" style="overflow-y:auto">
    <div class="content" id="headline-content" style="max-width:1320px"></div>
  </div>

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

  <!-- EXPERIMENTS (with sub-tabs) -->
  <div id="view-exp" class="view" style="flex-direction:column">
    <div class="subtabs">
      <div class="subtab active" id="subtab-mixing" onclick="expSub('mixing')">CPT-Mixing</div>
      <div class="subtab" id="subtab-docchunk" onclick="expSub('docchunk')">Doc-Chunked Attention</div>
    </div>
    <div id="exp-row" style="display:flex; flex:1; overflow:hidden">
      <div class="sidebar" id="exp-sidebar"></div>
      <div class="content" id="exp-content"></div>
    </div>
    <div class="content" id="docchunk-content" style="display:none; max-width:1300px"></div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const EXPER = __EXPER_JSON__;
const MIXING = __MIXING_JSON__;
const DOCCHUNK = __DOCCHUNK_JSON__;
const HEADLINE = __HEADLINE_JSON__;
let HL_RUNG = (HEADLINE && HEADLINE.headline_rung) || '32k';
const ORDER_COLORS = {"O(N)":"var(--on)","O(N^2)":"var(--on2)","O(N^3+)":"var(--on3)"};

function esc(s){ return s==null ? '' : String(s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function fmtTok(n){ if(n==null) return '?'; if(n>=1e6) return (n/1e6).toFixed(n>=1e7?0:1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(n>=1e4?0:1)+'K'; return ''+n; }

function switchTab(name){
  document.querySelectorAll('.view-tab').forEach(t=>t.classList.toggle('active', t.dataset.view===name));
  document.getElementById('view-headline').classList.toggle('active', name==='headline');
  document.getElementById('view-overview').classList.toggle('active', name==='overview');
  document.getElementById('view-data').classList.toggle('active', name==='data');
  document.getElementById('view-exp').classList.toggle('active', name==='exp');
}
let selExpSub='mixing';
function expSub(name){
  selExpSub=name;
  document.getElementById('subtab-mixing').classList.toggle('active', name==='mixing');
  document.getElementById('subtab-docchunk').classList.toggle('active', name==='docchunk');
  document.getElementById('exp-row').style.display = (name==='mixing')?'flex':'none';
  document.getElementById('docchunk-content').style.display = (name==='docchunk')?'block':'none';
  if(name==='docchunk') renderDocChunk();
}

/* ---------------- Headline · 32k matrix ---------------- */
const HL_VCOLOR = {dense:'#4A90E2', landmark:'#fbbf24', compressive:'#9B59B6'};
function hlPlaceholder(row){
  if(row.suspect) return `<td class="hl-na" title="broken HF-export eval — see note">n/a</td>`;
  if(row.status==='running') return `<td class="muted" style="font-size:10px">running</td>`;
  return `<td class="muted">pending</td>`;
}
function hlCell(row, task){
  if(row.suspect) return hlPlaceholder(row);
  let v;
  if(row.status==='base'){
    // Base (pre-SFT) rows have a single short-context number, not a per-rung ladder.
    v = (row.cell||{})[task];
  } else {
    v = (row.ladder && row.ladder[task]) ? row.ladder[task][HL_RUNG] : null;
    if(v==null && HL_RUNG===((HEADLINE&&HEADLINE.headline_rung)||'32k')) v = (row.cell||{})[task];
  }
  if(v==null) return hlPlaceholder(row);
  return `<td style="background:${scoreColor(v)}">${Number(v).toFixed(3)}</td>`;
}
function hlRulerCell(row){
  if(row.suspect) return hlPlaceholder(row);
  const v = row.ruler;
  if(v==null) return hlPlaceholder(row);
  return `<td style="background:${scoreColor(v)}"><b>${Number(v).toFixed(3)}</b></td>`;
}
function hlLadder(row, ncols){
  const H = HEADLINE;
  const tasks = H.tasks;
  // Which rungs actually appear for this row?
  const rungs = H.rungs.filter(rg => tasks.some(t => (row.ladder[t]||{})[rg]!=null));
  if(!rungs.length) return `<td colspan="${ncols}" class="hl-ladder hl-note" style="padding:8px 14px">No per-rung ladder available for this run.</td>`;
  let h = `<td colspan="${ncols}" class="hl-ladder"><table class="lad"><tr><th class="rl">task \\ ctx</th>`
    + rungs.map(rg=>`<th>${esc(rg)}</th>`).join('') + `</tr>`;
  tasks.forEach(t=>{
    const lr = row.ladder[t]||{};
    if(!Object.keys(lr).length) return;
    h += `<tr><td class="rl">${esc(H.task_labels[t]||t)}</td>`
      + rungs.map(rg=>{const v=lr[rg]; return v==null?`<td class="muted">&mdash;</td>`:`<td style="background:${scoreColor(v)}">${v.toFixed(3)}</td>`;}).join('')
      + `</tr>`;
  });
  return h + `</table></td>`;
}
function buildHeadline(){
  const H = HEADLINE;
  const c = document.getElementById('headline-content');
  if(!H || !H.groups){ c.innerHTML = '<div class="muted">No headline data.</div>'; return; }
  const tasks = H.tasks, ncols = tasks.length + 2; // rowlabel + tasks + RULER
  let h = `<div class="hl-title">${esc(H.title)}</div>`
    + `<div class="hl-sub">{ dense · landmark · compressive } × { +CPT-mix · no-CPT }, fine-tuned at 32k scale</div>`
    + `<div class="desc" style="max-width:1180px">${H.caption}</div>`;
  // variant legend
  h += `<div class="legend-row" style="margin-bottom:10px">`
    + Object.entries(HL_VCOLOR).map(([k,col])=>`<span><b style="background:${col}"></b>${esc(k)}</span>`).join('')
    + `<span class="muted">&nbsp;|&nbsp; column = the <b>${esc(HL_RUNG)} rung</b> · click a row to expand its full per-rung ladder</span></div>`;
  // Rung selector: the matrix below re-renders every cell at the chosen context length.
  const availRungs = H.rungs.filter(rg => H.groups.some(g=>g.rows.some(row =>
    row.ladder && tasks.some(t=>(row.ladder[t]||{})[rg]!=null))));
  if(availRungs.length && availRungs.indexOf(HL_RUNG)<0) HL_RUNG = availRungs[availRungs.length-1];
  h += `<div class="hl-rungsel"><span class="rlbl">context rung:</span>`
    + availRungs.map(rg=>`<button class="rungbtn ${rg===HL_RUNG?'active':''}" onclick="hlSetRung('${esc(rg)}')">${esc(rg)}</button>`).join('')
    + `<span class="muted" style="margin-left:8px">(base rows show their single pre-SFT number at every rung)</span></div>`;
  h += `<table class="hl"><tr><th class="rl">model · CPT-mix · scale</th>`
    + tasks.map(t=>`<th>${esc(H.task_labels[t]||t)}<br><span class="muted" style="font-weight:400">${esc(HL_RUNG)}</span></th>`).join('')
    + `<th>RULER<br><span class="muted" style="font-weight:400">avg recall</span></th></tr>`;
  let rid = 0;
  H.groups.forEach(g=>{
    h += `<tr class="grp"><td colspan="${ncols}">${esc(g.group)}</td></tr>`;
    g.rows.forEach(row=>{
      const id = 'hl'+(rid++);
      const hasLadder = row.ladder && Object.keys(row.ladder).length;
      const stBadge = row.suspect
        ? `<span class="hl-st suspect">eval n/a</span>`
        : ((row.status==='running'||row.status==='partial'||row.status==='base')
            ? `<span class="hl-st ${row.status}">${row.status}</span>` : '');
      const caret = hasLadder ? `<span class="hl-exp" id="${id}-car">&#9656; ladder</span>` : '';
      h += `<tr class="modelrow ${hasLadder?'expandable':''}" ${hasLadder?`onclick="hlToggle('${id}')"`:''}>`
        + `<td class="rl"><span class="hl-vdot" style="background:${HL_VCOLOR[row.variant]||'var(--dim)'}"></span>`
        + `<b>${esc(row.label)}</b>${stBadge}${caret}`
        + `<span class="hl-rawname">${esc(row.raw.startsWith('__')?'(not yet run)':row.raw)}</span></td>`
        + tasks.map(t=>hlCell(row,t)).join('')
        + hlRulerCell(row) + `</tr>`;
      if(hasLadder){
        h += `<tr class="ladrow" id="${id}-lad" style="display:none">` + hlLadder(row, ncols) + `</tr>`;
      }
      if(row.note){
        h += `<tr><td colspan="${ncols}" class="hl-note" style="padding:2px 12px 6px">&#9432; ${esc(row.note)}</td></tr>`;
      }
    });
  });
  h += `</table>`;
  if(H.footnotes && H.footnotes.length){
    h += `<div class="hl-foot"><ul>` + H.footnotes.map(f=>`<li>${esc(f)}</li>`).join('') + `</ul>`
      + `<div>Source: <code>${esc(H.source||'')}</code></div></div>`;
  }
  c.innerHTML = h;
}
function hlSetRung(rg){ HL_RUNG = rg; buildHeadline(); }
function hlToggle(id){
  const lad = document.getElementById(id+'-lad'); const car = document.getElementById(id+'-car');
  if(!lad) return;
  const open = lad.style.display!=='none';
  lad.style.display = open ? 'none' : '';
  if(car) car.innerHTML = (open?'&#9656;':'&#9662;') + ' ladder';
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
  box.innerHTML = `<div class="section-title">${esc(r.label)} &middot; ${r.examples.length} example(s)`
    + `<span class="hl-rawname" style="display:inline; margin-left:6px" title="source data file">${esc(r.file)}</span></div>`
    + r.examples.map(renderExample).join('');
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
function pct(v){ return v==null ? '<span class="muted">&mdash;</span>' : v+'%'; }
function num(v,d){ return v==null ? '<span class="muted">&mdash;</span>' : Number(v).toFixed(d); }

/* ---- Hypothesis-organized sections ---- */
const DASH = '<span class="muted">&mdash;</span>';
const COLDEFS = {
  model:    {h:'model',    text:r=>r.model},
  proxy:    {h:'tokens',   text:r=>r.proxy||'full'},
  cpt:      {h:'CPT%',     text:r=>r.cpt==null?DASH:r.cpt+'%'},
  lr:       {h:'LR',       text:r=>r.lr||DASH},
  ruler:    {h:'RULER',    kind:'ruler'},
  contra_f1:{h:'contra&nbsp;F1', kind:'score'},
  nq_f1:    {h:'NQ&nbsp;F1',     kind:'score'},
  oolong:   {h:'oolong',   kind:'score'},
  rerank:   {h:'rerank',   kind:'score'},
  outlier:  {h:'outlier',  kind:'score'},
  gen_fiqa: {h:'fiqa',     kind:'score'},
  gen_scifact:{h:'scifact',kind:'score'},
  gen_msmarco:{h:'msmarco',kind:'score'},
  gen_hpqa: {h:'hpqa',     kind:'score'},
  parse_rate:{h:'parse',   kind:'score'},
};
function rulerOk(){ return (MIXING && MIXING.ruler_ok!=null) ? MIXING.ruler_ok : 0.80; }
function renderCell(r,c){
  const cd = COLDEFS[c] || {h:c, kind:'score'};
  if(cd.text){ return `<td>${cd.text(r)}</td>`; }
  const v = r[c];
  if(cd.kind==='ruler'){
    if(v==null) return `<td>${DASH}</td>`;
    const deg = v < rulerOk();
    return `<td class="${deg?'degraded':''}" style="background:${scoreColor(v)}">`
      + `${v.toFixed(3)}${deg?'<span class="flag">&#9660;</span>':''}</td>`;
  }
  if(v==null) return `<td>${DASH}</td>`;
  return `<td style="background:${scoreColor(v)}">${Number(v).toFixed(3)}</td>`;
}
function hypoSections(){
  const m = MIXING;
  if(!m || !m.sections || !m.sections.length) return '';
  const byName = {}; (m.rows||[]).forEach(r=>{ byName[r.name]=r; });
  const ref = (m.ruler_ref!=null)? m.ruler_ref : 0.82;
  let h = `<div class="refbar">RULER reference (base CPT, no degradation) &asymp; <b>${ref}</b>. `
    + `Cells with RULER &lt; ${rulerOk()} are flagged <span class="flag">&#9660;</span> (degraded). `
    + `Scores are F1 / primary metric, 0&ndash;1; color = red&rarr;green.</div>`;
  m.sections.forEach(sec=>{
    h += `<div class="hsec"><div class="hsec-title">${esc(sec.title)}</div>`;
    h += `<div class="hyp"><b>Hypothesis:</b> ${esc(sec.hypothesis)}</div>`;
    const cols = sec.columns || ['ruler'];
    h += `<table><tr><th class="rowlabel">run</th>`
      + cols.map(c=>`<th>${(COLDEFS[c]||{h:c}).h}</th>`).join('') + `</tr>`;
    sec.runs.forEach(nm=>{
      const r = byName[nm];
      if(!r) return;
      const lbl = `${esc(r.disp||r.name)}${r.is_base?' <span class="muted">(no SFT)</span>':''}`
        + `<span class="hl-rawname">${esc(r.name)}</span>`;
      h += `<tr><td class="rowlabel" title="${esc(r.name)}">${lbl}</td>`
        + cols.map(c=>renderCell(r,c)).join('') + `</tr>`;
    });
    h += `</table>`;
    h += `<div class="take"><b>Takeaway:</b> ${esc(sec.takeaway)}</div></div>`;
  });
  return h;
}
function mixingTable(){
  const m = MIXING;
  if(!m || !m.rows || !m.rows.length) return '';
  let h = `<div class="section-title">CPT + T&uuml;lu + contradiction mixing experiments</div>`;
  if(m.caption) h += `<div class="desc" style="max-width:1100px">${m.caption}</div>`;
  h += `<table><tr>`
    + `<th class="rowlabel">run</th><th>model</th><th>CPT%</th><th>T&uuml;lu%</th>`
    + `<th>contra%</th><th>NQ%</th><th>LR</th><th>batch</th><th>RULER avg recall</th>`
    + `<th>&Delta;RULER vs base</th><th>contra F1</th><th>NQ F1</th><th>parse rate</th></tr>`;
  m.rows.forEach(r=>{
    const isBase = r.is_base;
    const label = (esc(r.disp||r.name)+(isBase?' <span class="muted">(no SFT)</span>':''))
      + `<span class="hl-rawname">${esc(r.name)}</span>`;
    let dr = '<span class="muted">&mdash;</span>';
    if(r.d_ruler!=null){
      const col = r.d_ruler>=0 ? 'var(--on)' : 'var(--on3)';
      const sign = r.d_ruler>=0 ? '+' : '';
      dr = `<span style="color:${col};font-weight:600">${sign}${r.d_ruler.toFixed(3)}</span>`;
    }
    h += `<tr><td class="rowlabel">${label}</td>`
      + `<td>${esc(r.model)}${r.proxy?' <span class="muted">('+esc(r.proxy)+')</span>':''}</td>`
      + `<td>${pct(r.cpt)}</td><td>${pct(r.tulu)}</td><td>${pct(r.contra)}</td><td>${pct(r.nq)}</td>`
      + `<td>${r.lr?esc(r.lr):'<span class="muted">&mdash;</span>'}</td>`
      + `<td>${r.batch?esc(r.batch):'<span class="muted">&mdash;</span>'}</td>`
      + `<td style="background:${scoreColor(r.ruler)}">${num(r.ruler,3)}</td>`
      + `<td>${dr}</td>`
      + `<td style="background:${scoreColor(r.contra_f1)}">${num(r.contra_f1,3)}</td>`
      + `<td style="background:${scoreColor(r.nq_f1)}">${num(r.nq_f1,3)}</td>`
      + `<td>${num(r.parse_rate,2)}</td></tr>`;
  });
  h += `</table><div class="note">&Delta;RULER is vs the same model's <code>*-cpt-base</code> row. `
    + `Source: ${esc(m.source||'')}.</div>`;
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
  let head = `<div class="detail-title">Contradiction CPT-mix — results by hypothesis</div>`
    + `<div class="desc">${esc(EXPER.narrative)}</div>`
    + hypoSections()
    + `<div class="section-title" style="margin-top:24px">Appendix — all runs</div>`
    + mixingTable() + rulerTable();
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

/* ---------------- Doc-Chunked Attention ---------------- */
const DC_CHUNK_COLORS = ['#4A90E2','#27AE60','#E67E22','#9B59B6','#16A085','#E74C3C'];
let selDcVariant = 'landmark';
function dcTokStyle(t){
  if(t.role==='ctx'||t.role==='marker'){
    const c = DC_CHUNK_COLORS[(t.chunk||0)%DC_CHUNK_COLORS.length];
    return t.role==='ctx' ? `background:${c};border-color:${c}` : `border-color:${c};color:${c}`;
  }
  return '';
}
function dcTokens(v){
  return `<div class="tokstrip">` + v.tokens.map(t=>
    `<span class="tok ${t.role}" style="${dcTokStyle(t)}" title="pos ${t.pos} &middot; ${t.role}${t.chunk!=null?' &middot; doc '+t.chunk:''}">`
    + `<span class="ix">${t.pos}</span>${esc(t.label)}</span>`).join('') + `</div>`;
}
function dcLegend(){
  const items = [['#1e293b','','FREE (instruction / query / answer — bridges all)'],
                 [DC_CHUNK_COLORS[0],'','document <i>i</i> (context chunk — isolated)'],
                 ['transparent','var(--accent)','&lt;box_start/end&gt; boundary'],
                 ['var(--highlight)','','&lt;landmark&gt; (block-end memory)'],
                 ['var(--surface)','','&lt;pad&gt; (window fill — non-attendable)']];
  return `<div class="dclegend">` + items.map(it=>
    `<span><b style="background:${it[0]}${it[1]?';border:1px solid '+it[1]:''}"></b>${it[2]}</span>`).join('') + `</div>`;
}
function dcMask(v){
  const M=v.mask, n=M.length, mem=v.is_mem||[];
  let h = `<div class="maskwrap"><table class="mask"><tr><th></th>`;
  for(let j=0;j<n;j++){ h+=`<th class="colh${mem[j]?' memcol':''}">${esc(v.tokens[j].label)}</th>`; }
  h+=`</tr>`;
  for(let i=0;i<n;i++){
    h+=`<tr class="${mem[i]?'memrow':''}"><th class="rowh">${esc(v.tokens[i].label)}</th>`;
    for(let j=0;j<n;j++){
      const on=M[i][j]===1;
      h+=`<td class="cell ${on?'on':'off'}${i===j?' diag':''}" title="q${i} (${esc(v.tokens[i].label)}) → k${j} (${esc(v.tokens[j].label)}): ${on?'attend':'blocked'}"></td>`;
    }
    h+=`</tr>`;
  }
  return h+`</table></div>`;
}
function dcResults(){
  const R=DOCCHUNK.results, gk=['user','timeline','counting'];
  let h=`<table><tr><th class="rowlabel">variant</th><th>OOLONG score</th><th>exact&nbsp;match</th>`
    + gk.map(g=>`<th>${g}</th>`).join('')+`</tr>`;
  R.rows.forEach(r=>{
    h+=`<tr><td class="rowlabel">${esc(r.label)}`
      + (r.out?`<span class="hl-rawname" title="eval result file">${esc(r.out)}</span>`:'')
      + `</td>`
      +`<td style="background:${scoreColor(r.score)}">${r.score.toFixed(3)}</td>`
      +`<td style="background:${scoreColor(r.em)}">${r.em.toFixed(3)}</td>`
      + gk.map(g=>{const x=(r.groups||{})[g]; return `<td style="background:${x!=null?scoreColor(x):'var(--surface)'}">${x!=null?x.toFixed(2):DASH}</td>`;}).join('')
      +`</tr>`;
  });
  return h+`</table><div class="note">OOLONG ${esc(R.ctx)}, eval_size=${R.eval_size}${R.eval_size<500?' <b style="color:#c00">&#9888; under 500 eval examples \u2014 differences here are within noise</b>':''}. Qwen3-4B, 3 epochs, native olmo_core eval (custom attention can't use HF/vLLM). Landmark ≈ dense while compressing each document to its window's memory.</div>`;
}
function selectDcVariant(v){ selDcVariant=v; renderDocChunk(); }
function renderDocChunk(){
  const c=document.getElementById('docchunk-content');
  const v=DOCCHUNK[selDcVariant];
  let h=`<div class="detail-title">Document-Chunked Attention</div>`
    +`<div class="desc" style="max-width:1120px">Each OOLONG <b>item line is one document</b> (so the document count grows with context length). Context documents are mutually <b>isolated</b>; the instruction / query / answer are <b>FREE</b> and bridge across every document. Boundaries are registered Qwen3 special tokens <code>&lt;box_start&gt;</code>/<code>&lt;box_end&gt;</code> (151648/151649). The <b>dense</b> variant applies this mask to ordinary attention; the <b>landmark</b> variant first-fit packs documents into landmark windows (block = mem_freq+1) with a memory token at every block end, and the FREE query retrieves each document through its window's landmark.</div>`;
  h+=`<div class="section-title">OOLONG results (dense vs. landmark)</div>`+dcResults();
  h+=`<div class="note" style="max-width:1100px"><b>Efficiency.</b> Dense uses a block-sparse `
    +`FlexAttention kernel at long context (numerically identical to the dense mask; 3&ndash;5&times; `
    +`faster at T&ge;16k, lower memory), and the materialized mask below 8k where it is faster. `
    +`Landmark inference supports hard <b>top-k</b> document retrieval &mdash; top-25% matches the `
    +`exact score (0.537 vs 0.536) while attending a quarter of the blocks.</div>`;
  h+=`<div class="section-title">What the model sees — tokens &amp; attention mask</div>`;
  h+=`<div class="dc-toggle">`
    +`<div class="opt ${selDcVariant==='dense'?'active':''}" onclick="selectDcVariant('dense')">dense</div>`
    +`<div class="opt ${selDcVariant==='landmark'?'active':''}" onclick="selectDcVariant('landmark')">landmark · mem_freq=${DOCCHUNK.landmark.mem_freq}</div>`
    +`</div>`;
  h+=`<div class="dc-cap">A tiny synthetic example built through the real <code>olmo_core.data.document_chunk_landmark</code> primitives: instruction (FREE) + 3 short documents + query (FREE).`
    + (selDcVariant==='landmark'
        ? ` Note the <b>packing</b>: document&nbsp;A shares window&nbsp;0 with the instruction, document&nbsp;C shares the last window with the query, and document&nbsp;B's window is padded — a <span style="color:var(--highlight)">landmark</span> sits at every block end (highlighted row/column).`
        : ` Documents are mutually blocked; FREE rows/columns attend everywhere causally.`)
    + `</div>`;
  h+=dcTokens(v)+dcLegend();
  h+=`<div class="section-title">Attention mask &mdash; row = query, column = key, <span style="color:var(--on)">green = attends</span></div>`+dcMask(v);
  c.innerHTML=h;
}

/* ---------------- init ---------------- */
buildHeadline();
buildOverview();
buildDataSidebar();
buildExpSidebar();
renderDocChunk();
if(DATA.tasks.length) selectTask(DATA.tasks[0].key);
if(EXPER.experiments.length) selectExp(EXPER.experiments[0].key);
</script>
</body>
</html>"""


def render(
    data_examples: dict, experiments: dict, mixing: dict, docchunk: dict, headline: dict
) -> str:
    stats = data_examples.get("stats", {})
    meta = (
        f"{stats.get('n_tasks', 0)} tasks &middot; "
        f"{stats.get('n_rungs', 0)} ladder rungs &middot; "
        f"{len(experiments.get('experiments', []))} CPT-mix experiments"
    )
    html = TEMPLATE
    html = html.replace("__TITLE__", "Corpus-Reasoning · Long-Context Suite & CPT-Mixing")
    html = html.replace("__META__", meta)
    html = html.replace("__DATA_JSON__", safe_json(data_examples))
    html = html.replace("__EXPER_JSON__", safe_json(experiments))
    html = html.replace("__MIXING_JSON__", safe_json(mixing))
    html = html.replace("__DOCCHUNK_JSON__", safe_json(docchunk))
    html = html.replace("__HEADLINE_JSON__", safe_json(headline))
    return html


def main():
    config.ensure_out_dir()
    with open(config.DATA_EXAMPLES_JSON) as f:
        data_examples = json.load(f)
    with open(config.EXPERIMENTS_JSON) as f:
        experiments = json.load(f)
    mixing_path = os.path.join(config.OUT_DIR, "mixing_results.json")
    mixing = {}
    if os.path.exists(mixing_path):
        with open(mixing_path) as f:
            mixing = json.load(f)
    docchunk_path = os.path.join(config.OUT_DIR, "docchunk.json")
    docchunk = {}
    if os.path.exists(docchunk_path):
        with open(docchunk_path) as f:
            docchunk = json.load(f)
    headline_path = os.path.join(config.OUT_DIR, "headline.json")
    headline = {}
    if os.path.exists(headline_path):
        with open(headline_path) as f:
            headline = json.load(f)
    html = render(data_examples, experiments, mixing, docchunk, headline)
    with open(config.SITE_HTML, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(config.SITE_HTML) / 1e6
    print(f"[render] wrote {config.SITE_HTML} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
