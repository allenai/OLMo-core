"""Web-based visualization for gradient analysis and representation comparison.

Serves an interactive HTML page with:
- Heatmap: x-axis = tokens, y-axis = layers, color = gradient magnitude or L2 distance
- Hover to see token text and exact values
- Toggle between examples
- Filter by token type (instruction/document/query/output)
- Support for both gradient analysis and representation comparison JSON files

Usage:
    python scripts/visualize_analysis.py \
        --gradient outputs/gradient_analysis.json \
        --comparison outputs/repr_comparison.json \
        --port 8080
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<title>Model Analysis Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }
.header { padding: 16px 24px; background: #16213e; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 20px; }
.header h1 { font-size: 18px; font-weight: 600; }
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.controls select, .controls button { padding: 6px 12px; border-radius: 4px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 13px; cursor: pointer; }
.controls select:hover, .controls button:hover { background: #1a5276; }
.controls button.active { background: #e94560; border-color: #e94560; }
.controls label { font-size: 13px; color: #aaa; }
.main { display: flex; height: calc(100vh - 56px); }
.sidebar { width: 280px; padding: 16px; overflow-y: auto; border-right: 1px solid #333; background: #16213e; }
.sidebar h3 { font-size: 14px; margin-bottom: 8px; color: #aaa; }
.stat { margin-bottom: 4px; font-size: 13px; }
.stat .label { color: #888; }
.stat .value { color: #eee; font-weight: 500; }
.legend { margin-top: 16px; }
.legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; }
.legend-color { width: 14px; height: 14px; border-radius: 2px; }
.canvas-container { flex: 1; overflow: auto; padding: 16px; position: relative; }
canvas { cursor: crosshair; }
.tooltip { position: fixed; background: rgba(0,0,0,0.9); border: 1px solid #555; border-radius: 6px; padding: 10px 14px; font-size: 12px; pointer-events: none; z-index: 1000; max-width: 400px; display: none; }
.tooltip .tt-token { font-weight: bold; color: #e94560; font-size: 14px; }
.tooltip .tt-type { color: #aaa; }
.tooltip .tt-value { color: #4fc3f7; margin-top: 4px; }
.token-strip { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 1px; max-height: 120px; overflow-y: auto; padding: 8px; background: #0f0f23; border-radius: 4px; }
.token-chip { padding: 2px 4px; font-size: 11px; border-radius: 2px; cursor: pointer; white-space: pre; }
.token-chip.instruction { background: #1a3a5c; }
.token-chip.document { background: #2d4a22; }
.token-chip.query { background: #5c3a1a; }
.token-chip.output { background: #4a224a; }
.token-chip.highlight { outline: 2px solid #e94560; }
.summary-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
.summary-table th, .summary-table td { padding: 4px 8px; text-align: left; border-bottom: 1px solid #333; }
.summary-table th { color: #888; font-weight: 500; }
</style>
</head>
<body>

<div class="header">
    <h1>Model Analysis</h1>
    <div class="controls">
        <label>File:</label>
        <select id="fileSelect"></select>
        <label>Example:</label>
        <select id="exampleSelect"></select>
        <label>Metric:</label>
        <select id="metricSelect">
            <option value="gradient_norms">Gradient Norms</option>
            <option value="l2_distances">L2 Distance</option>
            <option value="cosine_distances">Cosine Distance</option>
        </select>
        <label>Scale:</label>
        <select id="scaleSelect">
            <option value="linear">Linear</option>
            <option value="log">Log</option>
            <option value="percentile">Percentile</option>
        </select>
        <label>Filter:</label>
        <button class="type-filter active" data-type="instruction">Instr</button>
        <button class="type-filter active" data-type="document">Doc</button>
        <button class="type-filter active" data-type="query">Query</button>
        <button class="type-filter active" data-type="output">Output</button>
    </div>
</div>

<div class="main">
    <div class="sidebar">
        <h3>Info</h3>
        <div id="infoPanel"></div>
        <div class="legend">
            <h3>Token Types</h3>
            <div class="legend-item"><div class="legend-color" style="background:#1a3a5c"></div> Instruction</div>
            <div class="legend-item"><div class="legend-color" style="background:#2d4a22"></div> Document</div>
            <div class="legend-item"><div class="legend-color" style="background:#5c3a1a"></div> Query</div>
            <div class="legend-item"><div class="legend-color" style="background:#4a224a"></div> Output</div>
        </div>
        <div id="summaryPanel" style="margin-top:16px"></div>
        <h3 style="margin-top:16px">Tokens</h3>
        <div class="token-strip" id="tokenStrip"></div>
    </div>
    <div class="canvas-container">
        <canvas id="heatmap"></canvas>
    </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;

let currentFile = null;
let currentExample = null;
let activeTypes = new Set(['instruction', 'document', 'query', 'output']);
let highlightToken = -1;

// Init file select
const fileSelect = document.getElementById('fileSelect');
const exampleSelect = document.getElementById('exampleSelect');
const metricSelect = document.getElementById('metricSelect');
const scaleSelect = document.getElementById('scaleSelect');

Object.keys(DATA).forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.text = name;
    fileSelect.appendChild(opt);
});

fileSelect.addEventListener('change', () => { loadFile(fileSelect.value); });
exampleSelect.addEventListener('change', () => { loadExample(parseInt(exampleSelect.value)); });
metricSelect.addEventListener('change', render);
scaleSelect.addEventListener('change', render);

document.querySelectorAll('.type-filter').forEach(btn => {
    btn.addEventListener('click', () => {
        const t = btn.dataset.type;
        if (activeTypes.has(t)) { activeTypes.delete(t); btn.classList.remove('active'); }
        else { activeTypes.add(t); btn.classList.add('active'); }
        render();
    });
});

function loadFile(name) {
    currentFile = DATA[name];
    exampleSelect.innerHTML = '';
    currentFile.results.forEach((r, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.text = `Example ${r.example_idx} (${r.num_tokens} tokens)`;
        exampleSelect.appendChild(opt);
    });

    // Update available metrics
    const r0 = currentFile.results[0];
    metricSelect.innerHTML = '';
    if (r0.gradient_norms) {
        metricSelect.innerHTML += '<option value="gradient_norms">Gradient Norms</option>';
    }
    if (r0.l2_distances) {
        metricSelect.innerHTML += '<option value="l2_distances">L2 Distance</option>';
    }
    if (r0.cosine_distances) {
        metricSelect.innerHTML += '<option value="cosine_distances">Cosine Distance</option>';
    }

    loadExample(0);
}

function loadExample(idx) {
    currentExample = currentFile.results[idx];
    updateInfo();
    updateTokenStrip();
    render();
}

function updateInfo() {
    const e = currentExample;
    let html = '';
    html += `<div class="stat"><span class="label">Tokens:</span> <span class="value">${e.num_tokens}</span></div>`;
    html += `<div class="stat"><span class="label">Attention:</span> <span class="value">${e.attention_type}</span></div>`;
    html += `<div class="stat"><span class="label">Query pos:</span> <span class="value">${e.query_position}</span></div>`;
    if (e.loss !== undefined) {
        html += `<div class="stat"><span class="label">Loss:</span> <span class="value">${e.loss.toFixed(4)}</span></div>`;
    }
    if (currentFile.model_a) {
        html += `<div class="stat"><span class="label">Model A:</span> <span class="value">${currentFile.model_a}</span></div>`;
        html += `<div class="stat"><span class="label">Model B:</span> <span class="value">${currentFile.model_b}</span></div>`;
    }
    document.getElementById('infoPanel').innerHTML = html;

    // Summary table by token type
    const metric = metricSelect.value;
    const data = e[metric];
    if (!data) return;

    const layers = Object.keys(data).map(Number).sort((a,b) => a-b);
    const types = ['instruction', 'document', 'query', 'output'];
    let thtml = '<table class="summary-table"><tr><th>Type</th><th>Tokens</th><th>Avg</th><th>Max</th></tr>';
    types.forEach(ttype => {
        const indices = [];
        e.token_types.forEach((t, i) => { if (t === ttype) indices.push(i); });
        if (indices.length === 0) return;
        let sum = 0, max = 0, count = 0;
        layers.forEach(l => {
            indices.forEach(i => {
                const v = data[l][i];
                sum += v;
                if (v > max) max = v;
                count++;
            });
        });
        const avg = count > 0 ? sum / count : 0;
        thtml += `<tr><td>${ttype}</td><td>${indices.length}</td><td>${avg.toFixed(4)}</td><td>${max.toFixed(4)}</td></tr>`;
    });
    thtml += '</table>';
    document.getElementById('summaryPanel').innerHTML = thtml;
}

function updateTokenStrip() {
    const strip = document.getElementById('tokenStrip');
    strip.innerHTML = '';
    currentExample.tokens.forEach((tok, i) => {
        const chip = document.createElement('span');
        chip.className = 'token-chip ' + currentExample.token_types[i];
        chip.textContent = tok.replace('\n', '\\n');
        chip.addEventListener('mouseenter', () => { highlightToken = i; render(); });
        chip.addEventListener('mouseleave', () => { highlightToken = -1; render(); });
        strip.appendChild(chip);
    });
}

function render() {
    const canvas = document.getElementById('heatmap');
    const ctx = canvas.getContext('2d');

    if (!currentExample) return;

    const metric = metricSelect.value;
    const data = currentExample[metric];
    if (!data) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }

    const scale = scaleSelect.value;
    const layers = Object.keys(data).map(Number).sort((a, b) => a - b);
    const numTokens = currentExample.num_tokens;

    // Filter tokens
    const visibleTokens = [];
    currentExample.token_types.forEach((t, i) => {
        if (activeTypes.has(t)) visibleTokens.push(i);
    });

    if (visibleTokens.length === 0 || layers.length === 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }

    const cellW = Math.max(3, Math.min(12, Math.floor(1400 / visibleTokens.length)));
    const cellH = Math.max(12, Math.min(24, Math.floor(600 / layers.length)));
    const marginLeft = 60;
    const marginTop = 20;

    canvas.width = marginLeft + visibleTokens.length * cellW + 20;
    canvas.height = marginTop + layers.length * cellH + 40;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Collect all values for normalization
    const allVals = [];
    layers.forEach(l => {
        visibleTokens.forEach(ti => {
            const v = data[l][ti];
            if (v !== undefined && v > 0) allVals.push(v);
        });
    });

    if (allVals.length === 0) return;

    let normalize;
    if (scale === 'log') {
        const maxLog = Math.log1p(Math.max(...allVals));
        normalize = v => v > 0 ? Math.log1p(v) / maxLog : 0;
    } else if (scale === 'percentile') {
        const sorted = [...allVals].sort((a, b) => a - b);
        normalize = v => {
            let idx = sorted.findIndex(s => s >= v);
            if (idx === -1) idx = sorted.length;
            return idx / sorted.length;
        };
    } else {
        const maxVal = Math.max(...allVals);
        normalize = v => maxVal > 0 ? v / maxVal : 0;
    }

    // Draw heatmap
    layers.forEach((l, li) => {
        // Layer label
        ctx.fillStyle = '#888';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`L${l}`, marginLeft - 6, marginTop + li * cellH + cellH / 2 + 4);

        visibleTokens.forEach((ti, vi) => {
            const v = data[l][ti] || 0;
            const t = normalize(v);

            // Color: dark blue -> yellow -> red
            let r, g, b;
            if (t < 0.5) {
                const s = t * 2;
                r = Math.floor(10 + s * 245);
                g = Math.floor(10 + s * 200);
                b = Math.floor(80 - s * 60);
            } else {
                const s = (t - 0.5) * 2;
                r = Math.floor(255);
                g = Math.floor(210 - s * 180);
                b = Math.floor(20 - s * 20);
            }

            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(marginLeft + vi * cellW, marginTop + li * cellH, cellW - 1, cellH - 1);

            // Highlight column
            if (ti === highlightToken) {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.strokeRect(marginLeft + vi * cellW, marginTop + li * cellH, cellW - 1, cellH - 1);
            }
        });
    });

    // Mouse handler for tooltip
    canvas.onmousemove = (evt) => {
        const rect = canvas.getBoundingClientRect();
        const x = evt.clientX - rect.left;
        const y = evt.clientY - rect.top;

        const vi = Math.floor((x - marginLeft) / cellW);
        const li = Math.floor((y - marginTop) / cellH);

        const tooltip = document.getElementById('tooltip');
        if (vi >= 0 && vi < visibleTokens.length && li >= 0 && li < layers.length) {
            const ti = visibleTokens[vi];
            const l = layers[li];
            const v = data[l][ti] || 0;
            const tok = currentExample.tokens[ti];
            const ttype = currentExample.token_types[ti];

            tooltip.innerHTML = `
                <div class="tt-token">"${tok.replace(/</g, '&lt;').replace('\n', '\\n')}"</div>
                <div class="tt-type">Token ${ti} | ${ttype} | Layer ${l}</div>
                <div class="tt-value">${metric}: ${v.toFixed(6)}</div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (evt.clientX + 16) + 'px';
            tooltip.style.top = (evt.clientY + 16) + 'px';
        } else {
            tooltip.style.display = 'none';
        }
    };
    canvas.onmouseleave = () => {
        document.getElementById('tooltip').style.display = 'none';
    };
}

// Init
if (Object.keys(DATA).length > 0) {
    loadFile(Object.keys(DATA)[0]);
}
</script>
</body>
</html>"""


class AnalysisHandler(BaseHTTPRequestHandler):
    data_json = "{}"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", self.data_json)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        elif parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(self.data_json.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description="Visualize gradient/representation analysis")
    parser.add_argument("--gradient", type=str, nargs="*", default=[],
                        help="Gradient analysis JSON files")
    parser.add_argument("--comparison", type=str, nargs="*", default=[],
                        help="Representation comparison JSON files")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--save-html", type=str, default="",
                        help="Save standalone HTML file instead of serving")
    args = parser.parse_args()

    # Load all data files
    all_data = {}
    for path in args.gradient:
        name = Path(path).stem
        with open(path) as f:
            all_data[f"grad: {name}"] = json.load(f)
    for path in args.comparison:
        name = Path(path).stem
        with open(path) as f:
            all_data[f"repr: {name}"] = json.load(f)

    if not all_data:
        print("No data files specified. Use --gradient and/or --comparison flags.")
        return

    data_json = json.dumps(all_data)
    print(f"Loaded {len(all_data)} analysis file(s)")

    if args.save_html:
        html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
        Path(args.save_html).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_html, "w") as f:
            f.write(html)
        print(f"Saved standalone HTML to {args.save_html}")
        return

    AnalysisHandler.data_json = data_json
    server = HTTPServer(("0.0.0.0", args.port), AnalysisHandler)
    print(f"Serving at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
