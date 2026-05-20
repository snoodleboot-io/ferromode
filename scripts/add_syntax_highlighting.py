#!/usr/bin/env python3
"""
Apply syntax-highlighting spans to new interop language tabs in examples.html.
Targets tab-content divs for: javascript, julia, r, cpp, matlab, grpc.
Uses the same custom span classes as the hand-crafted Python/Rust/CLI blocks:
  .keyword  .string  .comment  .number  .function  .type
"""

import re

# ---------------------------------------------------------------------------
# Per-language token rules  (applied in order — first match wins per segment)
# ---------------------------------------------------------------------------

COMMENT   = 'comment'
STRING    = 'string'
KEYWORD   = 'keyword'
FUNCTION  = 'function'
NUMBER    = 'number'
TYPE      = 'type'

JS_KEYWORDS = (
    r'import|export|from|as|await|async|const|let|var|new|for|of|in|if|else|'
    r'return|function|class|true|false|null|undefined|this'
)
JS_TYPES = r'Float64Array|Promise|Array|Map|Set|Error'

JULIA_KEYWORDS = (
    r'using|import|export|function|end|for|in|if|else|elseif|return|'
    r'while|begin|struct|mutable|module|let|local|global|const|true|false|nothing'
)
JULIA_TYPES = r'Float64|Int|UInt|Bool|String|Vector|Matrix|Dict|Ref|Ptr|Cvoid|Cint|Cdouble|Csize_t|Culonglong'

R_KEYWORDS = (
    r'library|require|for|in|if|else|function|return|while|repeat|break|next|'
    r'TRUE|FALSE|NULL|NA|Inf|NaN'
)
R_FUNCTIONS = r'cat|sprintf|paste|print|scan|read\.table|load|sum|mean|var|Reduce|sapply|rowSums|seq_along|length|numel|list'

CPP_KEYWORDS = (
    r'int|double|float|size_t|long|short|unsigned|signed|bool|void|char|'
    r'for|while|if|else|return|auto|const|static|struct|class|new|delete|'
    r'namespace|using|include|define|ifdef|endif|true|false|nullptr|this'
)
CPP_TYPES = r'std::vector|std::string|std::ifstream|std::istringstream|std::cout|std::accumulate|std::getline|size_t'

MATLAB_KEYWORDS = (
    r'for|end|if|else|elseif|while|function|return|break|continue|'
    r'true|false|load|save|fprintf|disp|numel|length|size|sum|mean|var|'
    r'zeros|ones|eye|linspace|sqrt|abs|real|imag|cat'
)

GRPC_KEYWORDS = (
    r'import|from|as|with|for|in|if|else|return|def|class|lambda|'
    r'True|False|None|and|or|not|print|list|sum|len|enumerate|range|zip'
)
GRPC_TYPES = r'grpc|np|pb|pb_grpc'


LANG_RULES = {
    'javascript': [
        (COMMENT,   r'//[^\n]*'),
        (STRING,    r"'(?:[^'\\]|\\.)*'"),
        (STRING,    r'`(?:[^`\\]|\\.)*`'),
        (TYPE,      r'\b(?:' + JS_TYPES + r')\b'),
        (KEYWORD,   r'\b(?:' + JS_KEYWORDS + r')\b'),
        (FUNCTION,  r'\b([a-zA-Z_$][\w$]*)(?=\s*\()'),
        (NUMBER,    r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
    ],
    'julia': [
        (COMMENT,   r'#[^\n]*'),
        (STRING,    r'"(?:[^"\\]|\\.)*"'),
        (TYPE,      r'\b(?:' + JULIA_TYPES + r')\b'),
        (KEYWORD,   r'\b(?:' + JULIA_KEYWORDS + r')\b'),
        (FUNCTION,  r'\b([a-zA-Z_][\w!]*)(?=\s*\()'),
        (NUMBER,    r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
    ],
    'r': [
        (COMMENT,   r'#[^\n]*'),
        (STRING,    r'"(?:[^"\\]|\\.)*"'),
        (STRING,    r"'(?:[^'\\]|\\.)*'"),
        (KEYWORD,   r'\b(?:' + R_KEYWORDS + r')\b'),
        (FUNCTION,  r'\b(?:' + R_FUNCTIONS + r')(?=\s*\()'),
        (FUNCTION,  r'\b([a-zA-Z_.][\w.]*)(?=\s*\()'),
        (NUMBER,    r'\b\d+(?:\.\d+)?(?:L|e[+-]?\d+)?\b'),
    ],
    'cpp': [
        (COMMENT,   r'//[^\n]*'),
        (STRING,    r'"(?:[^"\\]|\\.)*"'),
        (TYPE,      r'\b(?:' + CPP_TYPES.replace('|', r'(?:::\w+)*|') + r')\b'),
        (KEYWORD,   r'\b(?:' + CPP_KEYWORDS + r')\b'),
        (FUNCTION,  r'\b([a-zA-Z_][\w]*)(?=\s*\()'),
        (NUMBER,    r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
    ],
    'matlab': [
        (COMMENT,   r'%[^\n]*'),
        (STRING,    r"'(?:[^'\n])*'"),
        (KEYWORD,   r'\b(?:' + MATLAB_KEYWORDS + r')\b'),
        (FUNCTION,  r'\b([a-zA-Z_][\w]*)(?=\s*\()'),
        (NUMBER,    r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
    ],
    'grpc': [
        (COMMENT,   r'#[^\n]*'),
        (STRING,    r'"(?:[^"\\]|\\.)*"'),
        (STRING,    r"'(?:[^'\\]|\\.)*'"),
        (STRING,    r'f"(?:[^"\\]|\\.)*"'),
        (STRING,    r"f'(?:[^'\\]|\\.)*'"),
        (TYPE,      r'\b(?:' + GRPC_TYPES + r')\b'),
        (KEYWORD,   r'\b(?:' + GRPC_KEYWORDS + r')\b'),
        (FUNCTION,  r'\b([a-zA-Z_][\w]*)(?=\s*\()'),
        (NUMBER,    r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
    ],
}

CLASS_MAP = {
    COMMENT:  'comment',
    STRING:   'string',
    KEYWORD:  'keyword',
    FUNCTION: 'function',
    NUMBER:   'number',
    TYPE:     'type',
}


def highlight(code_html: str, lang: str) -> str:
    """Apply span-based syntax highlighting to HTML-escaped code text."""
    rules = LANG_RULES.get(lang, [])
    if not rules:
        return code_html

    # Build a single combined regex with named groups.
    # Each rule gets a unique group name like g0, g1, ...
    parts = []
    for i, (token_type, pattern) in enumerate(rules):
        parts.append(f'(?P<g{i}>{pattern})')
    combined = re.compile('|'.join(parts))

    def replace(m):
        # Find which group matched
        for i, (token_type, _) in enumerate(rules):
            text = m.group(f'g{i}')
            if text is not None:
                css = CLASS_MAP[token_type]
                return f'<span class="{css}">{text}</span>'
        return m.group(0)

    return combined.sub(replace, code_html)


def process_file(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()

    target_langs = set(LANG_RULES.keys())

    # Match each tab-content div for our target languages.
    # Pattern: <div class="tab-content" data-lang="LANG">...<code>CODE</code></div>
    # We only want to modify blocks that have NO existing spans (new blocks).
    tab_pattern = re.compile(
        r'(<div class="tab-content" data-lang="('
        + '|'.join(re.escape(l) for l in sorted(target_langs))
        + r')">\s*<div class="code-block">\s*'
        + r'<span class="code-badge">[^<]*</span>'
        + r'<span class="code-copy-btn"[^>]*>[^<]*</span>'
        + r'<code>)(.*?)(</code></div>\s*</div>)',
        re.DOTALL,
    )

    def replace_tab(m):
        prefix  = m.group(1)
        lang    = m.group(2)
        code    = m.group(3)
        suffix  = m.group(4)

        # Skip blocks that already have span highlighting
        if '<span class=' in code:
            return m.group(0)

        highlighted = highlight(code, lang)
        return prefix + highlighted + suffix

    new_html = tab_pattern.sub(replace_tab, html)

    changed = new_html.count('<span class=') - html.count('<span class=')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_html)

    print(f"Added {changed} highlight spans across new interop tabs.")


if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, '..', 'webpage', 'examples.html')
    process_file(target)
