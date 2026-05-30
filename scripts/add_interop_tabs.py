#!/usr/bin/env python3
"""Add 6 new language tabs (JavaScript/WASM, Julia, R, C++, MATLAB, gRPC) to all 9
example groups in webpage/examples.html."""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
HTML_FILE = ROOT / "webpage" / "examples.html"

# ── Step 1: button replacement ────────────────────────────────────────────────

NEW_BUTTONS = """\
            <button class="tab-btn" data-lang="cli" >CLI</button>
            <button class="tab-btn" data-lang="javascript">JavaScript</button>
            <button class="tab-btn" data-lang="julia">Julia</button>
            <button class="tab-btn" data-lang="r">R</button>
            <button class="tab-btn" data-lang="cpp">C++</button>
            <button class="tab-btn" data-lang="matlab">MATLAB</button>
            <button class="tab-btn" data-lang="grpc">gRPC</button>"""

OLD_BUTTON = '            <button class="tab-btn" data-lang="cli" >CLI</button>'

# ── Tab-content blocks keyed by unique CLI output path ────────────────────────

def make_block(lang, badge, code):
    """Return a fully-formatted tab-content div."""
    return (
        f'          <div class="tab-content" data-lang="{lang}">\n'
        f'            <div class="code-block">\n'
        f'              <span class="code-badge">{badge}</span>'
        f'<span class="code-copy-btn" onclick="copyCode(this)">Copy</span>'
        f'<code>{code}</code></div>\n'
        f'          </div>'
    )


# ── Per-example content definitions ──────────────────────────────────────────

EXAMPLES = {

# ── 1: ECG ────────────────────────────────────────────────────────────────────
"imfs/ecg_imfs.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const resp = await fetch('data/ecg_sample.csv');
const signal = new Float64Array((await resp.text()).trim().split('\\n').map(Number));

const config = new WasmEmdConfig(0.2, 1, 100, 5,
  WasmBoundaryCondition.ExtremasMirror, true, 1e-12);
const result = emd_wasm(signal, config);

console.log(`Extracted ${result.imfs().n_imfs()} IMFs`);
for (let i = 0; i &lt; result.imfs().n_imfs(); i++) {
  const imf = result.imfs().get_imf(i);
  const energy = imf.reduce((s, v) =&gt; s + v * v, 0);
  console.log(`  IMF ${i + 1}: energy = ${energy.toFixed(4)}`);
}"""),
    ("julia", "Julia", """\
using Ferromode

signal = parse.(Float64, readlines("data/ecg_sample.csv"))
result = Ferromode.emd(signal; max_imfs=5)

println("Extracted $(result.n_imfs) IMFs in $(round(result.elapsed_ms, digits=1)) ms")
for i in 0:(result.n_imfs - 1)
    imf    = Ferromode.get_imf(result, i)
    energy = sum(v^2 for v in imf)
    println("  IMF $(i + 1): energy = $(round(energy, digits=4))")
end
residue = Ferromode.get_residue(result)
clean   = sum(Ferromode.get_imf(result, i) for i in 1:(result.n_imfs - 1)) .+ residue"""),
    ("r", "R", """\
library(ferrormodeR)

signal &lt;- scan("data/ecg_sample.csv", quiet = TRUE)
result &lt;- emd_decompose(signal, max_imfs = 5L)

cat(sprintf("Extracted %d IMFs\\n", length(result$imfs)))
for (i in seq_along(result$imfs)) {
  energy &lt;- sum(result$imfs[[i]]^2)
  cat(sprintf("  IMF %d: energy = %.4f\\n", i, energy))
}
# Reconstruct without IMF 1 (50 Hz noise)
clean &lt;- Reduce("+", result$imfs[-1]) + result$residue"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;fstream&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  std::ifstream f("data/ecg_sample.csv");
  std::vector&lt;double&gt; signal;
  double v;
  while (f &gt;&gt; v) signal.push_back(v);

  ferromode::EmdConfig cfg{};
  cfg.max_imfs     = 5;
  cfg.sd_threshold = 0.2;

  auto r = ferromode::ferromode_emd_cxx(signal.data(), signal.size(), &amp;cfg);
  std::cout &lt;&lt; "Extracted " &lt;&lt; r.n_imfs &lt;&lt; " IMFs\\n";
  for (size_t i = 0; i &lt; r.n_imfs; ++i) {
    double e = 0;
    for (size_t j = 0; j &lt; r.n_samples; ++j) {
      double x = r.imfs_data[i * r.n_samples + j]; e += x * x;
    }
    std::cout &lt;&lt; "  IMF " &lt;&lt; (i+1) &lt;&lt; ": energy = " &lt;&lt; e &lt;&lt; "\\n";
  }
}"""),
    ("matlab", "MATLAB", """\
signal = load('data/ecg_sample.csv');

cfg.max_imfs     = 5;
cfg.sd_threshold = 0.2;

result = ferromode_mex('emd', signal, cfg);
fprintf('Extracted %d IMFs\\n', numel(result.imfs));

for i = 1:numel(result.imfs)
    energy = sum(result.imfs{i}.^2);
    fprintf('  IMF %d: energy = %.4f\\n', i, energy);
end
% Drop IMF 1 (50 Hz artifact), reconstruct clean cardiac signal
clean = sum(cat(2, result.imfs{2:end}), 2) + result.residue;"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

signal = np.loadtxt('data/ecg_sample.csv')

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=5),
    ))

print(f"Extracted {len(resp.imfs)} IMFs")
for i, imf in enumerate(resp.imfs):
    energy = sum(v**2 for v in imf.values)
    print(f"  IMF {i+1}: energy = {energy:.4f}")"""),
],

# ── 2: Seismic ────────────────────────────────────────────────────────────────
"imfs/seismic_imfs.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const resp = await fetch('data/seismic_sample.csv');
const signal = new Float64Array((await resp.text()).trim().split('\\n').map(Number));

const config = new WasmEmdConfig(0.2, 1, 100, 5,
  WasmBoundaryCondition.ExtremasMirror, true, 1e-12);
const result = emd_wasm(signal, config);

const fs = 100.0;
console.log(`Separated ${result.imfs().n_imfs()} wave components`);
for (let i = 0; i &lt; result.imfs().n_imfs(); i++) {
  const imf = result.imfs().get_imf(i);
  const energy = imf.reduce((s, v) =&gt; s + v * v, 0);
  console.log(`  IMF ${i + 1}: energy = ${energy.toFixed(4)}`);
}"""),
    ("julia", "Julia", """\
using Ferromode

signal = parse.(Float64, readlines("data/seismic_sample.csv"))
result = Ferromode.emd(signal; max_imfs=5)

println("Separated $(result.n_imfs) wave components")
for i in 0:(result.n_imfs - 1)
    imf    = Ferromode.get_imf(result, i)
    energy = sum(v^2 for v in imf)
    println("  IMF $(i + 1): energy = $(round(energy, sigdigits=4))")
end"""),
    ("r", "R", """\
library(ferrormodeR)

signal &lt;- scan("data/seismic_sample.csv", quiet = TRUE)
result &lt;- emd_decompose(signal, max_imfs = 5L)

cat(sprintf("Separated %d wave components\\n", length(result$imfs)))
for (i in seq_along(result$imfs)) {
  energy &lt;- sum(result$imfs[[i]]^2)
  cat(sprintf("  IMF %d: energy = %.4f\\n", i, energy))
}"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;fstream&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  std::ifstream f("data/seismic_sample.csv");
  std::vector&lt;double&gt; signal;
  double v;
  while (f &gt;&gt; v) signal.push_back(v);

  ferromode::EmdConfig cfg{};
  cfg.max_imfs = 5;

  auto r = ferromode::ferromode_emd_cxx(signal.data(), signal.size(), &amp;cfg);
  std::cout &lt;&lt; "Separated " &lt;&lt; r.n_imfs &lt;&lt; " wave components\\n";
  for (size_t i = 0; i &lt; r.n_imfs; ++i) {
    double e = 0;
    for (size_t j = 0; j &lt; r.n_samples; ++j) {
      double x = r.imfs_data[i * r.n_samples + j]; e += x * x;
    }
    std::cout &lt;&lt; "  IMF " &lt;&lt; (i+1) &lt;&lt; ": energy = " &lt;&lt; e &lt;&lt; "\\n";
  }
}"""),
    ("matlab", "MATLAB", """\
signal = load('data/seismic_sample.csv');

cfg.max_imfs = 5;

result = ferromode_mex('emd', signal, cfg);
fprintf('Separated %d wave components\\n', numel(result.imfs));

for i = 1:numel(result.imfs)
    energy = sum(result.imfs{i}.^2);
    fprintf('  IMF %d: energy = %.4f\\n', i, energy);
end"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

signal = np.loadtxt('data/seismic_sample.csv')

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=5),
    ))

print(f"Separated {len(resp.imfs)} wave components")
for i, imf in enumerate(resp.imfs):
    energy = sum(v**2 for v in imf.values)
    print(f"  IMF {i+1}: energy = {energy:.4f}")"""),
],

# ── 3: Speech ─────────────────────────────────────────────────────────────────
"imfs/speech_imfs.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const resp = await fetch('data/speech_sample.csv');
const signal = new Float64Array((await resp.text()).trim().split('\\n').map(Number));

const config = new WasmEmdConfig(0.15, 1, 100, 4,
  WasmBoundaryCondition.ExtremasMirror, true, 1e-12);
const result = emd_wasm(signal, config);

console.log(`Extracted ${result.imfs().n_imfs()} IMFs`);
// Drop IMF 1 (burst noise), reconstruct clean speech
let clean = result.imfs().get_residue();
for (let i = 1; i &lt; result.imfs().n_imfs(); i++) {
  const imf = result.imfs().get_imf(i);
  clean = clean.map((v, j) =&gt; v + imf[j]);
}"""),
    ("julia", "Julia", """\
using Ferromode

signal = parse.(Float64, readlines("data/speech_sample.csv"))
result = Ferromode.emd(signal; max_imfs=4, sd_threshold=0.15)

println("Extracted $(result.n_imfs) IMFs")
residue = Ferromode.get_residue(result)
imf0    = Ferromode.get_imf(result, 0)
noise_energy = sum(v^2 for v in imf0)
clean = residue .+ sum(Ferromode.get_imf(result, i) for i in 1:(result.n_imfs - 1))
println("Removed noise energy: $(round(noise_energy, sigdigits=4))")"""),
    ("r", "R", """\
library(ferrormodeR)

signal &lt;- scan("data/speech_sample.csv", quiet = TRUE)
result &lt;- emd_decompose(signal, max_imfs = 4L)

cat(sprintf("Extracted %d IMFs\\n", length(result$imfs)))
noise_energy &lt;- sum(result$imfs[[1]]^2)
cat(sprintf("Removed noise energy: %.4f\\n", noise_energy))

# Drop IMF 1 (burst noise), reconstruct clean speech
clean &lt;- Reduce("+", result$imfs[-1]) + result$residue"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;fstream&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  std::ifstream f("data/speech_sample.csv");
  std::vector&lt;double&gt; signal;
  double v;
  while (f &gt;&gt; v) signal.push_back(v);

  ferromode::EmdConfig cfg{};
  cfg.max_imfs     = 4;
  cfg.sd_threshold = 0.15;

  auto r = ferromode::ferromode_emd_cxx(signal.data(), signal.size(), &amp;cfg);
  std::cout &lt;&lt; "Extracted " &lt;&lt; r.n_imfs &lt;&lt; " IMFs\\n";
  double noise_energy = 0;
  for (size_t j = 0; j &lt; r.n_samples; ++j) {
    double x = r.imfs_data[j]; noise_energy += x * x;
  }
  std::cout &lt;&lt; "Removed noise energy: " &lt;&lt; noise_energy &lt;&lt; "\\n";
}"""),
    ("matlab", "MATLAB", """\
signal = load('data/speech_sample.csv');

cfg.max_imfs     = 4;
cfg.sd_threshold = 0.15;

result = ferromode_mex('emd', signal, cfg);
fprintf('Extracted %d IMFs\\n', numel(result.imfs));
noise_energy = sum(result.imfs{1}.^2);
fprintf('Removed noise energy: %.4f\\n', noise_energy);

% Reconstruct clean speech (drop IMF 1)
clean = sum(cat(2, result.imfs{2:end}), 2) + result.residue;"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

signal = np.loadtxt('data/speech_sample.csv')

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=4),
    ))

imfs = [list(imf.values) for imf in resp.imfs]
noise_energy = sum(v**2 for v in imfs[0])
clean = np.array(imfs[1:]).sum(axis=0) + np.array(list(resp.residue.values))
print(f"Removed noise energy: {noise_energy:.4f}")"""),
],

# ── 4: Finance ────────────────────────────────────────────────────────────────
"imfs/aapl_imfs.csv": [
    ("javascript", "JavaScript", """\
import init, { iceemdan_wasm, WasmEmdConfig, WasmEnsembleConfig,
               WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const resp = await fetch('data/aapl_log_returns.csv');
const signal = new Float64Array((await resp.text()).trim().split('\\n').map(Number));

const emdCfg = new WasmEmdConfig(0.005, 1, 100, 6,
  WasmBoundaryCondition.ExtremasMirror, true, 1e-12);
const ensCfg = new WasmEnsembleConfig(100, 0.2, 0, true);
const result = iceemdan_wasm(signal, ensCfg, emdCfg);

console.log(`Extracted ${result.imfs().n_imfs()} IMFs`);
for (let i = 0; i &lt; result.imfs().n_imfs(); i++) {
  const imf = result.imfs().get_imf(i);
  const variance = imf.reduce((s, v) =&gt; s + v * v, 0) / imf.length;
  console.log(`  IMF ${i + 1}: variance = ${variance.toFixed(6)}`);
}"""),
    ("julia", "Julia", """\
using Ferromode

signal = parse.(Float64, readlines("data/aapl_log_returns.csv"))
result = Ferromode.iceemdan(signal;
    num_ensembles=100, noise_std=0.2, seed=0,
    max_imfs=6, sd_threshold=0.005)

println("Extracted $(result.n_imfs) IMFs")
total_var = sum(v^2 for v in signal) / length(signal)
for i in 0:(result.n_imfs - 1)
    imf    = Ferromode.get_imf(result, i)
    var_pct = (sum(v^2 for v in imf) / length(imf)) / total_var * 100
    println("  IMF $(i+1): $(round(var_pct, digits=1))% of total variance")
end"""),
    ("r", "R", """\
library(ferrormodeR)

signal &lt;- scan("data/aapl_log_returns.csv", quiet = TRUE)

ens_cfg &lt;- list(num_ensembles = 100L, noise_std = 0.2, seed = 0L)
emd_cfg &lt;- list(max_imfs = 6L, sd_threshold = 0.005)
result  &lt;- iceemdan(signal, ens_cfg, emd_cfg)

cat(sprintf("Extracted %d IMFs\\n", length(result$imfs)))
total_var &lt;- var(signal)
for (i in seq_along(result$imfs)) {
  var_pct &lt;- var(result$imfs[[i]]) / total_var * 100
  cat(sprintf("  IMF %d: %.1f%% of total variance\\n", i, var_pct))
}"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;fstream&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  std::ifstream f("data/aapl_log_returns.csv");
  std::vector&lt;double&gt; signal;
  double v;
  while (f &gt;&gt; v) signal.push_back(v);

  ferromode::EnsembleConfig ens{};
  ens.num_ensembles = 100;
  ens.noise_std     = 0.2;
  ens.seed          = 0;
  ens.use_seed      = true;

  ferromode::EmdConfig cfg{};
  cfg.max_imfs     = 6;
  cfg.sd_threshold = 0.005;

  auto r = ferromode::ferromode_iceemdan_cxx(
      signal.data(), signal.size(), &amp;ens, &amp;cfg);
  std::cout &lt;&lt; "Extracted " &lt;&lt; r.n_imfs &lt;&lt; " IMFs\\n";
}"""),
    ("matlab", "MATLAB", """\
signal = load('data/aapl_log_returns.csv');

ens_cfg.num_ensembles = 100;
ens_cfg.noise_std     = 0.2;
ens_cfg.seed          = 0;
emd_cfg.max_imfs      = 6;
emd_cfg.sd_threshold  = 0.005;

result = ferromode_mex('iceemdan', signal, ens_cfg, emd_cfg);
fprintf('Extracted %d IMFs\\n', numel(result.imfs));

total_var = var(signal);
for i = 1:numel(result.imfs)
    var_pct = var(result.imfs{i}) / total_var * 100;
    fprintf('  IMF %d: %.1f%% of total variance\\n', i, var_pct);
end"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

signal = np.loadtxt('data/aapl_log_returns.csv')

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=6, algorithm='iceemdan'),
    ))

print(f"Extracted {len(resp.imfs)} IMFs")
total_var = float(np.var(signal))
for i, imf in enumerate(resp.imfs):
    vals = np.array(imf.values)
    print(f"  IMF {i+1}: {np.var(vals)/total_var*100:.1f}% of total variance")"""),
],

# ── 5: ExtremasMirror ─────────────────────────────────────────────────────────
"imfs/extrema.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const fs = 256, n = 1024;
const signal = new Float64Array(n);
for (let i = 0; i &lt; n; i++) {
  const t = i / fs;
  signal[i] = [32,16,8,2].reduce((s,f) =&gt; s + Math.sin(2*Math.PI*f*t), 0);
}

const config = new WasmEmdConfig(0.2, 1, 100, 4,
  WasmBoundaryCondition.ExtremasMirror, true, 1e-12);
const result = emd_wasm(signal, config);
console.log(`IMFs: ${result.imfs().n_imfs()}`);"""),
    ("julia", "Julia", """\
using Ferromode

fs, n = 256.0, 1024
signal = [sum(sin(2\u03c0 * f * i/fs) for f in [32,16,8,2]) for i in 0:(n-1)]
result = Ferromode.emd(signal; max_imfs=4)
println("IMFs: $(result.n_imfs)")"""),
    ("r", "R", """\
library(ferrormodeR)

fs &lt;- 256; n &lt;- 1024
t &lt;- (0:(n-1)) / fs
signal &lt;- rowSums(sapply(c(32,16,8,2), function(f) sin(2*pi*f*t)))
result &lt;- emd_decompose(signal, max_imfs = 4L)
cat(sprintf("IMFs: %d\\n", length(result$imfs)))"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;cmath&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  const int fs = 256, n = 1024;
  std::vector&lt;double&gt; signal(n);
  for (int i = 0; i &lt; n; ++i) {
    double t = (double)i / fs;
    for (double f : {32.0, 16.0, 8.0, 2.0})
      signal[i] += std::sin(2 * M_PI * f * t);
  }

  ferromode::EmdConfig cfg{};
  cfg.max_imfs = 4;
  // boundary_condition 0 = ExtremasMirror (default)
  auto r = ferromode::ferromode_emd_cxx(signal.data(), n, &amp;cfg);
  std::cout &lt;&lt; "IMFs: " &lt;&lt; r.n_imfs &lt;&lt; "\\n";
}"""),
    ("matlab", "MATLAB", """\
fs = 256; n = 1024;
t  = (0:n-1) / fs;
signal = sum(sin(2*pi*[32;16;8;2] .* t), 1)';

cfg.max_imfs = 4;
result = ferromode_mex('emd', signal, cfg);
fprintf('IMFs: %d\\n', numel(result.imfs));"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

fs, n = 256, 1024
t = np.arange(n) / fs
signal = sum(np.sin(2*np.pi*f*t) for f in [32, 16, 8, 2])

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=4),
    ))
print(f"IMFs: {len(resp.imfs)}")"""),
],

# ── 6: MirrorEven ─────────────────────────────────────────────────────────────
"imfs/mirror.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const fs = 256, n = 1024;
const signal = new Float64Array(n);
for (let i = 0; i &lt; n; i++) {
  const t = i / fs;
  signal[i] = [32,16,8,2].reduce((s,f) =&gt; s + Math.sin(2*Math.PI*f*t), 0);
}

const config = new WasmEmdConfig(0.2, 1, 100, 4,
  WasmBoundaryCondition.MirrorEven, true, 1e-12);
const result = emd_wasm(signal, config);
console.log(`IMFs: ${result.imfs().n_imfs()}`);"""),
    ("julia", "Julia", """\
using Ferromode

fs, n = 256.0, 1024
signal = [sum(sin(2\u03c0 * f * i/fs) for f in [32,16,8,2]) for i in 0:(n-1)]
# boundary_condition 2 = MirrorEven
result = Ferromode.emd(signal; max_imfs=4, boundary_condition=2)
println("IMFs: $(result.n_imfs)")"""),
    ("r", "R", """\
library(ferrormodeR)

fs &lt;- 256; n &lt;- 1024
t &lt;- (0:(n-1)) / fs
signal &lt;- rowSums(sapply(c(32,16,8,2), function(f) sin(2*pi*f*t)))
result &lt;- emd_decompose(signal, max_imfs = 4L)
# Note: boundary condition configuration coming in a future R API update
cat(sprintf("IMFs: %d\\n", length(result$imfs)))"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;cmath&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  const int fs = 256, n = 1024;
  std::vector&lt;double&gt; signal(n);
  for (int i = 0; i &lt; n; ++i) {
    double t = (double)i / fs;
    for (double f : {32.0, 16.0, 8.0, 2.0})
      signal[i] += std::sin(2 * M_PI * f * t);
  }

  ferromode::EmdConfig cfg{};
  cfg.max_imfs          = 4;
  cfg.boundary_condition = 2;  // MirrorEven
  auto r = ferromode::ferromode_emd_cxx(signal.data(), n, &amp;cfg);
  std::cout &lt;&lt; "IMFs: " &lt;&lt; r.n_imfs &lt;&lt; "\\n";
}"""),
    ("matlab", "MATLAB", """\
fs = 256; n = 1024;
t  = (0:n-1) / fs;
signal = sum(sin(2*pi*[32;16;8;2] .* t), 1)';

cfg.max_imfs          = 4;
cfg.boundary_condition = 2;  % MirrorEven
result = ferromode_mex('emd', signal, cfg);
fprintf('IMFs: %d\\n', numel(result.imfs));"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

fs, n = 256, 1024
t = np.arange(n) / fs
signal = sum(np.sin(2*np.pi*f*t) for f in [32, 16, 8, 2])

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=4, boundary='mirror'),
    ))
print(f"IMFs: {len(resp.imfs)}")"""),
],

# ── 7: Periodic ───────────────────────────────────────────────────────────────
"imfs/periodic.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const fs = 256, n = 1024;
const signal = new Float64Array(n);
for (let i = 0; i &lt; n; i++) {
  const t = i / fs;
  signal[i] = [32,16,8,2].reduce((s,f) =&gt; s + Math.sin(2*Math.PI*f*t), 0);
}

const config = new WasmEmdConfig(0.2, 1, 100, 4,
  WasmBoundaryCondition.Periodic, true, 1e-12);
const result = emd_wasm(signal, config);
console.log(`IMFs: ${result.imfs().n_imfs()}`);"""),
    ("julia", "Julia", """\
using Ferromode

fs, n = 256.0, 1024
signal = [sum(sin(2\u03c0 * f * i/fs) for f in [32,16,8,2]) for i in 0:(n-1)]
# boundary_condition 4 = Periodic
result = Ferromode.emd(signal; max_imfs=4, boundary_condition=4)
println("IMFs: $(result.n_imfs)")"""),
    ("r", "R", """\
library(ferrormodeR)

fs &lt;- 256; n &lt;- 1024
t &lt;- (0:(n-1)) / fs
signal &lt;- rowSums(sapply(c(32,16,8,2), function(f) sin(2*pi*f*t)))
result &lt;- emd_decompose(signal, max_imfs = 4L)
cat(sprintf("IMFs: %d\\n", length(result$imfs)))"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;cmath&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  const int fs = 256, n = 1024;
  std::vector&lt;double&gt; signal(n);
  for (int i = 0; i &lt; n; ++i) {
    double t = (double)i / fs;
    for (double f : {32.0, 16.0, 8.0, 2.0})
      signal[i] += std::sin(2 * M_PI * f * t);
  }

  ferromode::EmdConfig cfg{};
  cfg.max_imfs           = 4;
  cfg.boundary_condition = 4;  // Periodic
  auto r = ferromode::ferromode_emd_cxx(signal.data(), n, &amp;cfg);
  std::cout &lt;&lt; "IMFs: " &lt;&lt; r.n_imfs &lt;&lt; "\\n";
}"""),
    ("matlab", "MATLAB", """\
fs = 256; n = 1024;
t  = (0:n-1) / fs;
signal = sum(sin(2*pi*[32;16;8;2] .* t), 1)';

cfg.max_imfs           = 4;
cfg.boundary_condition = 4;  % Periodic
result = ferromode_mex('emd', signal, cfg);
fprintf('IMFs: %d\\n', numel(result.imfs));"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

fs, n = 256, 1024
t = np.arange(n) / fs
signal = sum(np.sin(2*np.pi*f*t) for f in [32, 16, 8, 2])

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=4, boundary='periodic'),
    ))
print(f"IMFs: {len(resp.imfs)}")"""),
],

# ── 8: WaveformMatching ───────────────────────────────────────────────────────
"imfs/waveform.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const fs = 256, n = 1024;
const signal = new Float64Array(n);
for (let i = 0; i &lt; n; i++) {
  const t = i / fs;
  signal[i] = [32,16,8,2].reduce((s,f) =&gt; s + Math.sin(2*Math.PI*f*t), 0);
}

const config = new WasmEmdConfig(0.2, 1, 100, 4,
  WasmBoundaryCondition.WaveformMatching, true, 1e-12);
const result = emd_wasm(signal, config);
console.log(`IMFs: ${result.imfs().n_imfs()}`);"""),
    ("julia", "Julia", """\
using Ferromode

fs, n = 256.0, 1024
signal = [sum(sin(2\u03c0 * f * i/fs) for f in [32,16,8,2]) for i in 0:(n-1)]
# boundary_condition 7 = WaveformMatching
result = Ferromode.emd(signal; max_imfs=4, boundary_condition=7)
println("IMFs: $(result.n_imfs)")"""),
    ("r", "R", """\
library(ferrormodeR)

fs &lt;- 256; n &lt;- 1024
t &lt;- (0:(n-1)) / fs
signal &lt;- rowSums(sapply(c(32,16,8,2), function(f) sin(2*pi*f*t)))
result &lt;- emd_decompose(signal, max_imfs = 4L)
cat(sprintf("IMFs: %d\\n", length(result$imfs)))"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;cmath&gt;
#include &lt;iostream&gt;
#include &lt;vector&gt;

int main() {
  const int fs = 256, n = 1024;
  std::vector&lt;double&gt; signal(n);
  for (int i = 0; i &lt; n; ++i) {
    double t = (double)i / fs;
    for (double f : {32.0, 16.0, 8.0, 2.0})
      signal[i] += std::sin(2 * M_PI * f * t);
  }

  ferromode::EmdConfig cfg{};
  cfg.max_imfs           = 4;
  cfg.boundary_condition = 7;  // WaveformMatching
  auto r = ferromode::ferromode_emd_cxx(signal.data(), n, &amp;cfg);
  std::cout &lt;&lt; "IMFs: " &lt;&lt; r.n_imfs &lt;&lt; "\\n";
}"""),
    ("matlab", "MATLAB", """\
fs = 256; n = 1024;
t  = (0:n-1) / fs;
signal = sum(sin(2*pi*[32;16;8;2] .* t), 1)';

cfg.max_imfs           = 4;
cfg.boundary_condition = 7;  % WaveformMatching
result = ferromode_mex('emd', signal, cfg);
fprintf('IMFs: %d\\n', numel(result.imfs));"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

fs, n = 256, 1024
t = np.arange(n) / fs
signal = sum(np.sin(2*np.pi*f*t) for f in [32, 16, 8, 2])

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=4, boundary='waveform'),
    ))
print(f"IMFs: {len(resp.imfs)}")"""),
],

# ── 9: Sunspots ───────────────────────────────────────────────────────────────
"imfs/sunspots_imfs.csv": [
    ("javascript", "JavaScript", """\
import init, { emd_wasm, WasmEmdConfig, WasmBoundaryCondition } from './ferromode_wasm.js';
await init();

const resp = await fetch('data/sunspots_annual.txt');
const lines = (await resp.text()).trim().split('\\n');
const raw = new Float64Array(lines.map(l =&gt; +l.trim().split(/\\s+/)[1]));
const mean = raw.reduce((s, v) =&gt; s + v, 0) / raw.length;
const signal = raw.map(v =&gt; v - mean);

const config = new WasmEmdConfig(0.2, 1, 100, 4,
  WasmBoundaryCondition.Periodic, true, 1e-12);
const result = emd_wasm(signal, config);

console.log(`Extracted ${result.imfs().n_imfs()} IMFs`);
for (let i = 0; i &lt; result.imfs().n_imfs(); i++) {
  const imf = result.imfs().get_imf(i);
  const energy = imf.reduce((s, v) =&gt; s + v * v, 0);
  console.log(`  IMF ${i + 1}: energy = ${energy.toFixed(2)}`);
}"""),
    ("julia", "Julia", """\
using Ferromode

lines  = readlines("data/sunspots_annual.txt")
values = [parse(Float64, split(l)[2]) for l in lines]
signal = values .- sum(values) / length(values)  # de-mean

# boundary_condition 4 = Periodic
result = Ferromode.emd(signal; max_imfs=4, boundary_condition=4)

println("Extracted $(result.n_imfs) IMFs")
for i in 0:(result.n_imfs - 1)
    imf    = Ferromode.get_imf(result, i)
    energy = sum(v^2 for v in imf)
    println("  IMF $(i+1): energy = $(round(energy, sigdigits=4))")
end"""),
    ("r", "R", """\
library(ferrormodeR)

data   &lt;- read.table("data/sunspots_annual.txt", header = FALSE)
signal &lt;- data$V2 - mean(data$V2)

result &lt;- emd_decompose(signal, max_imfs = 4L)
cat(sprintf("Extracted %d IMFs\\n", length(result$imfs)))

for (i in seq_along(result$imfs)) {
  energy &lt;- sum(result$imfs[[i]]^2)
  cat(sprintf("  IMF %d: energy = %.2f\\n", i, energy))
}"""),
    ("cpp", "C++", """\
#include &lt;ferromode.hpp&gt;
#include &lt;fstream&gt;
#include &lt;iostream&gt;
#include &lt;numeric&gt;
#include &lt;sstream&gt;
#include &lt;vector&gt;

int main() {
  std::ifstream f("data/sunspots_annual.txt");
  std::vector&lt;double&gt; values;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    double year, val;
    if (ss &gt;&gt; year &gt;&gt; val) values.push_back(val);
  }
  double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  for (auto &amp;v : values) v -= mean;

  ferromode::EmdConfig cfg{};
  cfg.max_imfs           = 4;
  cfg.boundary_condition = 4;  // Periodic
  auto r = ferromode::ferromode_emd_cxx(values.data(), values.size(), &amp;cfg);
  std::cout &lt;&lt; "Extracted " &lt;&lt; r.n_imfs &lt;&lt; " IMFs\\n";
}"""),
    ("matlab", "MATLAB", """\
data   = load('data/sunspots_annual.txt');
signal = data(:,2) - mean(data(:,2));

cfg.max_imfs           = 4;
cfg.boundary_condition = 4;  % Periodic

result = ferromode_mex('emd', signal, cfg);
fprintf('Extracted %d IMFs\\n', numel(result.imfs));

for i = 1:numel(result.imfs)
    energy = sum(result.imfs{i}.^2);
    fprintf('  IMF %d: energy = %.2f\\n', i, energy);
end"""),
    ("grpc", "gRPC", """\
import grpc, numpy as np
import ferromode.v1.emd_pb2 as pb
import ferromode.v1.emd_pb2_grpc as pb_grpc

data   = np.loadtxt('data/sunspots_annual.txt')
signal = data[:, 1] - data[:, 1].mean()

with grpc.insecure_channel('localhost:50051') as ch:
    stub = pb_grpc.EmdServiceStub(ch)
    resp = stub.Decompose(pb.DecomposeRequest(
        signal=pb.Signal(values=signal.tolist()),
        config=pb.EmdConfig(max_imfs=4, boundary='periodic'),
    ))

print(f"Extracted {len(resp.imfs)} IMFs")
for i, imf in enumerate(resp.imfs):
    energy = sum(v**2 for v in imf.values)
    print(f"  IMF {i+1}: energy = {energy:.2f}")"""),
],

}  # end EXAMPLES


def build_new_tabs(tabs):
    """Build the 6 new tab-content divs as a single string (newline-separated)."""
    parts = []
    for lang, badge, code in tabs:
        parts.append(make_block(lang, badge, code))
    return "\n" + "\n".join(parts)


def main():
    html = HTML_FILE.read_text(encoding="utf-8")

    # ── Step 1: add 6 new tab buttons after every CLI button ──────────────────
    count_buttons = html.count(OLD_BUTTON)
    assert count_buttons == 9, f"Expected 9 CLI buttons, found {count_buttons}"
    html = html.replace(OLD_BUTTON, NEW_BUTTONS)

    # ── Step 2: insert tab-content blocks after each CLI block ────────────────
    CLOSING = "</code></div>\n          </div>"

    for unique_path, tabs in EXAMPLES.items():
        idx = html.find(unique_path)
        assert idx != -1, f"Could not find unique path: {unique_path}"

        # find the closing pattern AFTER the unique path
        close_idx = html.find(CLOSING, idx)
        assert close_idx != -1, f"Could not find closing tag after: {unique_path}"

        insert_pos = close_idx + len(CLOSING)
        new_content = build_new_tabs(tabs)
        html = html[:insert_pos] + new_content + html[insert_pos:]

    # ── Verify ────────────────────────────────────────────────────────────────
    js_count = html.count('data-lang="javascript"')
    print(f'data-lang="javascript" occurrences: {js_count}')
    assert js_count == 9 * 2, (  # button + tab-content = 2 per example
        f"Expected 18 javascript occurrences (9 buttons + 9 content), got {js_count}"
    )

    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"Successfully updated {HTML_FILE}")


if __name__ == "__main__":
    main()
