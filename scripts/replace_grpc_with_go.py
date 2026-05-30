"""Replace gRPC tabs with Go (ferromode-go) tabs in examples.html."""
import re

HTML = "webpage/examples.html"

with open(HTML, encoding="utf-8") as f:
    html = f.read()

# ---- Go code blocks for each example ----------------------------------
# Keyed by a unique substring that appears inside each gRPC block to
# identify which example we are in.

def hl(code: str, lang: str = "go") -> str:
    """Apply syntax highlighting spans to Go source code."""
    # HTML-escape first pass is already done in the source; we receive plain text
    # and will return highlighted HTML.
    go_keywords = [
        "package", "import", "func", "var", "const", "type", "struct", "interface",
        "return", "if", "else", "for", "range", "switch", "case", "default",
        "break", "continue", "goto", "defer", "go", "chan", "select",
        "map", "make", "new", "len", "cap", "append", "copy", "delete",
        "nil", "true", "false", "error", "panic", "recover",
    ]
    go_types = [
        "int", "int64", "int32", "uint64", "float64", "float32",
        "string", "bool", "byte", "rune", "[]float64", "[]byte",
        "EmdConfig", "EnsembleConfig", "VmdConfig", "Result",
    ]
    go_funcs = [
        r"ferromode\.\w+", r"result\.\w+", r"fmt\.\w+", r"math\.\w+",
        r"os\.\w+", r"bufio\.\w+", r"csv\.\w+", r"strconv\.\w+",
        r"log\.\w+",
    ]

    import html as html_mod

    lines = code.split("\n")
    result_lines = []
    for line in lines:
        # comment
        if "//" in line:
            idx = line.index("//")
            pre = line[:idx]
            comment = line[idx:]
            line = _highlight_line(pre, go_keywords, go_types) + f'<span class="comment">{html_mod.escape(comment)}</span>'
            result_lines.append(line)
            continue
        line = _highlight_line(line, go_keywords, go_types)
        result_lines.append(line)
    return "\n".join(result_lines)


import html as _html_mod

def _highlight_line(line: str, keywords: list, types: list) -> str:
    # We process raw text (not yet HTML-escaped) token by token.
    # Strategy: scan character by character, emit highlighted spans.
    # For simplicity use regex substitutions on the escaped line.
    escaped = _html_mod.escape(line)

    # strings (double-quoted)
    escaped = re.sub(r'(&quot;[^&]*?&quot;|"[^"]*?")', r'<span class="string">\1</span>', escaped)

    # backtick strings
    escaped = re.sub(r'(`[^`]*?`)', r'<span class="string">\1</span>', escaped)

    # numbers (standalone)
    escaped = re.sub(r'\b(\d+\.?\d*(?:e[+-]?\d+)?)\b', r'<span class="number">\1</span>', escaped)

    # ferromode package calls
    escaped = re.sub(r'\b(ferromode)\b', r'<span class="type">\1</span>', escaped)

    # function calls (identifier followed by open-paren, not already in a span)
    escaped = re.sub(r'\b([A-Za-z_]\w*)\s*(?=\()', r'<span class="function">\1</span>', escaped)

    # types
    for t in types:
        escaped = re.sub(r'\b(' + re.escape(t) + r')\b', r'<span class="type">\1</span>', escaped)

    # keywords
    for kw in keywords:
        escaped = re.sub(r'\b(' + re.escape(kw) + r')\b', r'<span class="keyword">\1</span>', escaped)

    return escaped


def make_go_block(badge: str, code: str) -> str:
    highlighted = hl(code)
    return (
        f'          <div class="tab-content" data-lang="go">\n'
        f'            <div class="code-block">\n'
        f'              <span class="code-badge">Go</span>'
        f'<span class="code-copy-btn" onclick="copyCode(this)">Copy</span>'
        f'<code>{highlighted}</code></div>\n'
        f'          </div>'
    )


# ----------- Go code per example ---------------------------------------

GO_ECG = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"os"
	"encoding/csv"
	"strconv"
	ferromode "github.com/ferromode/ferromode-go"
)

func loadCSV(path string) []float64 {
	f, _ := os.Open(path)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i], _ = strconv.ParseFloat(r[0], 64)
	}
	return out
}

func main() {
	signal := loadCSV("data/ecg_sample.csv")
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 5
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("Extracted %d IMFs\\n", result.NImfs)
	for i := 0; i < result.NImfs; i++ {
		imf, _ := result.IMF(i)
		var energy float64
		for _, v := range imf {
			energy += v * v
		}
		fmt.Printf("  IMF %d: energy = %.4f\\n", i+1, energy)
	}
}\
''')

GO_SEISMIC = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"os"
	"encoding/csv"
	"strconv"
	ferromode "github.com/ferromode/ferromode-go"
)

func loadCSV(path string) []float64 {
	f, _ := os.Open(path)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i], _ = strconv.ParseFloat(r[0], 64)
	}
	return out
}

func main() {
	signal := loadCSV("data/seismic_sample.csv")
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 5
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("Separated %d wave components\\n", result.NImfs)
	for i := 0; i < result.NImfs; i++ {
		imf, _ := result.IMF(i)
		var energy float64
		for _, v := range imf {
			energy += v * v
		}
		fmt.Printf("  IMF %d: energy = %.4f\\n", i+1, energy)
	}
}\
''')

GO_SPEECH = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"os"
	"encoding/csv"
	"strconv"
	ferromode "github.com/ferromode/ferromode-go"
)

func loadCSV(path string) []float64 {
	f, _ := os.Open(path)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i], _ = strconv.ParseFloat(r[0], 64)
	}
	return out
}

func main() {
	signal := loadCSV("data/speech_sample.csv")
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 4
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	noise, _ := result.IMF(0)
	var noiseEnergy float64
	for _, v := range noise {
		noiseEnergy += v * v
	}
	fmt.Printf("Removed noise energy: %.4f\\n", noiseEnergy)
}\
''')

GO_FINANCE = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"os"
	"encoding/csv"
	"strconv"
	ferromode "github.com/ferromode/ferromode-go"
)

func loadCSV(path string) []float64 {
	f, _ := os.Open(path)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i], _ = strconv.ParseFloat(r[0], 64)
	}
	return out
}

func main() {
	signal := loadCSV("data/aapl_log_returns.csv")
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 6
	ens := ferromode.DefaultEnsembleConfig()
	result, err := ferromode.ICEEMDAN(signal, ens, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	var totalVar float64
	for _, v := range signal {
		totalVar += v * v
	}
	fmt.Printf("Extracted %d IMFs\\n", result.NImfs)
	for i := 0; i < result.NImfs; i++ {
		imf, _ := result.IMF(i)
		var v float64
		for _, x := range imf {
			v += x * x
		}
		fmt.Printf("  IMF %d: %.1f%% of total variance\\n", i+1, v/totalVar*100)
	}
}\
''')

# Boundary examples — synthetic signal, varying BoundaryCondition
GO_BOUNDARY_EXTREMAS = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"math"
	ferromode "github.com/ferromode/ferromode-go"
)

func main() {
	const fs, n = 256.0, 1024
	signal := make([]float64, n)
	for i := range signal {
		t := float64(i) / fs
		for _, f := range []float64{32, 16, 8, 2} {
			signal[i] += math.Sin(2 * math.Pi * f * t)
		}
	}
	cfg := ferromode.DefaultEmdConfig() // BoundaryCondition 0 = ExtremasMirror
	cfg.MaxImfs = 4
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("IMFs: %d\\n", result.NImfs)
}\
''')

GO_BOUNDARY_MIRROREVEN = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"math"
	ferromode "github.com/ferromode/ferromode-go"
)

func main() {
	const fs, n = 256.0, 1024
	signal := make([]float64, n)
	for i := range signal {
		t := float64(i) / fs
		for _, f := range []float64{32, 16, 8, 2} {
			signal[i] += math.Sin(2 * math.Pi * f * t)
		}
	}
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 4
	cfg.BoundaryCondition = 1 // MirrorEven
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("IMFs: %d\\n", result.NImfs)
}\
''')

GO_BOUNDARY_PERIODIC = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"math"
	ferromode "github.com/ferromode/ferromode-go"
)

func main() {
	const fs, n = 256.0, 1024
	signal := make([]float64, n)
	for i := range signal {
		t := float64(i) / fs
		for _, f := range []float64{32, 16, 8, 2} {
			signal[i] += math.Sin(2 * math.Pi * f * t)
		}
	}
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 4
	cfg.BoundaryCondition = 2 // Periodic (Zhao-Huang)
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("IMFs: %d\\n", result.NImfs)
}\
''')

GO_BOUNDARY_WAVEFORM = make_go_block("Go", '''\
package main

import (
	"fmt"
	"log"
	"math"
	ferromode "github.com/ferromode/ferromode-go"
)

func main() {
	const fs, n = 256.0, 1024
	signal := make([]float64, n)
	for i := range signal {
		t := float64(i) / fs
		for _, f := range []float64{32, 16, 8, 2} {
			signal[i] += math.Sin(2 * math.Pi * f * t)
		}
	}
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 4
	cfg.BoundaryCondition = 6 // WaveformMatching
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("IMFs: %d\\n", result.NImfs)
}\
''')

GO_SUNSPOTS = make_go_block("Go", '''\
package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	ferromode "github.com/ferromode/ferromode-go"
)

func loadSunspots(path string) []float64 {
	f, _ := os.Open(path)
	defer f.Close()
	var vals []float64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		v, _ := strconv.ParseFloat(fields[1], 64)
		vals = append(vals, v)
	}
	return vals
}

func mean(s []float64) float64 {
	var sum float64
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func main() {
	raw := loadSunspots("data/sunspots_annual.txt")
	mu := mean(raw)
	signal := make([]float64, len(raw))
	for i, v := range raw {
		signal[i] = v - mu
	}
	cfg := ferromode.DefaultEmdConfig()
	cfg.MaxImfs = 4
	cfg.BoundaryCondition = 2 // Periodic
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer result.Free()
	fmt.Printf("Extracted %d IMFs\\n", result.NImfs)
	for i := 0; i < result.NImfs; i++ {
		imf, _ := result.IMF(i)
		var energy float64
		for _, v := range imf {
			energy += math.Pow(v, 2)
		}
		fmt.Printf("  IMF %d: energy = %.2f\\n", i+1, energy)
	}
}\
''')

# ---- Map each gRPC block to its replacement ---------------------------
# We identify each block by a unique string inside the gRPC code content.
REPLACEMENTS = [
    ("ecg_sample.csv", GO_ECG),
    ("seismic_sample.csv", GO_SEISMIC),
    ("speech_sample.csv", GO_SPEECH),
    ("aapl_log_returns.csv", GO_FINANCE),
    # boundary examples: identified by boundary= param or lack thereof
    # ExtremasMirror has no boundary kwarg and max_imfs=4 (first boundary block)
    ("EmdConfig(max_imfs=4),", GO_BOUNDARY_EXTREMAS),
    ("boundary=&#x27;mirror&#x27;", GO_BOUNDARY_MIRROREVEN),
    # Zhao-Huang periodic (before waveform in file order)
    ("boundary=&#x27;periodic&#x27;", None),  # two hits: periodic and sunspots
    ("boundary=&#x27;waveform&#x27;", GO_BOUNDARY_WAVEFORM),
    ("sunspots_annual.txt", GO_SUNSPOTS),
]

# The periodic boundary block appears TWICE (Zhao-Huang example + sunspots).
# We'll handle them separately via positional replacement.

# Pattern to match a full grpc tab-content div (non-greedy up to closing </div>)
GRPC_BLOCK_RE = re.compile(
    r'<div class="tab-content" data-lang="grpc">.*?</div>\s*(?=\n\s*(?:<div|$))',
    re.DOTALL,
)

blocks = list(GRPC_BLOCK_RE.finditer(html))
print(f"Found {len(blocks)} gRPC blocks")

# Build ordered list of replacements based on identifying content
ordered_go = []
for m in blocks:
    content = m.group()
    if "ecg_sample.csv" in content:
        ordered_go.append(GO_ECG)
    elif "seismic_sample.csv" in content:
        ordered_go.append(GO_SEISMIC)
    elif "speech_sample.csv" in content:
        ordered_go.append(GO_SPEECH)
    elif "aapl_log_returns.csv" in content:
        ordered_go.append(GO_FINANCE)
    elif "sunspots_annual.txt" in content:
        ordered_go.append(GO_SUNSPOTS)
    elif "boundary=&#x27;mirror&#x27;" in content or "boundary='mirror'" in content:
        ordered_go.append(GO_BOUNDARY_MIRROREVEN)
    elif "boundary=&#x27;waveform&#x27;" in content or "boundary='waveform'" in content:
        ordered_go.append(GO_BOUNDARY_WAVEFORM)
    elif "boundary=&#x27;periodic&#x27;" in content or "boundary='periodic'" in content:
        ordered_go.append(GO_BOUNDARY_PERIODIC)
    else:
        # ExtremasMirror block (no boundary kwarg)
        ordered_go.append(GO_BOUNDARY_EXTREMAS)

assert len(ordered_go) == len(blocks), f"{len(ordered_go)} vs {len(blocks)}"

# Replace from end to start to preserve offsets
new_html = html
for m, go_block in reversed(list(zip(blocks, ordered_go))):
    new_html = new_html[:m.start()] + go_block + "\n" + new_html[m.end():]

# Also replace the tab buttons: data-lang="grpc">gRPC</button>
new_html = new_html.replace(
    '<button class="tab-btn" data-lang="grpc">gRPC</button>',
    '<button class="tab-btn" data-lang="go">Go</button>',
)

with open(HTML, "w", encoding="utf-8") as f:
    f.write(new_html)

print("Done — gRPC tabs replaced with Go tabs")
