"""Structural checks for the manuscript, standing in for a compiler we do not have.

Catches the classes of error that would otherwise surface only as "??" in a rendered
PDF or as an outright LaTeX abort: unbalanced environments and braces, control
characters from mis-escaped macros, dangling cross-references, undefined citation keys,
and draft-mode leftovers.
"""
import collections
import io
import re
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex"
src = io.open(path, encoding="utf-8").read()
problems = 0


def report(label, bad, detail=""):
    global problems
    if bad:
        problems += 1
        print(f"  FAIL  {label}: {detail or bad}")
    else:
        print(f"  ok    {label}")


# 1. environments
envs = collections.Counter()
for m in re.finditer(r"\\(begin|end)\{([a-zA-Z*]+)\}", src):
    envs[m.group(2)] += 1 if m.group(1) == "begin" else -1
report("environments balanced", {k: v for k, v in envs.items() if v != 0})

# 2. braces, ignoring escaped ones
esc = re.sub(r"\\[{}]", "", src)
delta = esc.count("{") - esc.count("}")
report("braces balanced", delta != 0, f"delta {delta}")

# 3. control characters that should never reach a .tex file
ctrl = {repr(c): src.count(c) for c in "\a\b\f\v" if c in src}
report("no control characters", ctrl)

# 4. math mode: $ must be paired (ignore \$). Report the offending line, since a global
#    parity count tells you there is a problem but not where it is.
def count_math_dollars(line):
    """Count real math delimiters.

    A dollar is escaped only when preceded by an *odd* number of backslashes: in
    ``\\$X$`` the ``\\\\`` is a line break and the dollar that follows opens math, so a
    naive one-character lookbehind miscounts every TikZ node label in the paper.
    """
    n, i = 0, 0
    while i < len(line):
        if line[i] == "$":
            b = 0
            while i - 1 - b >= 0 and line[i - 1 - b] == chr(92):
                b += 1
            if b % 2 == 0:
                n += 1
                if i + 1 < len(line) and line[i + 1] == "$":
                    i += 1          # display-math delimiter counts once
        i += 1
    return n


odd_lines = []
open_math = False
for lineno, line in enumerate(src.split("\n"), 1):
    n = count_math_dollars(line)
    if n % 2:
        odd_lines.append((lineno, open_math, line.strip()[:70]))
        open_math = not open_math
report("inline math paired", open_math,
       f"math left open; {len(odd_lines)} odd-$ lines, first at "
       f"{odd_lines[0][0] if odd_lines else '-'}")
if open_math and odd_lines:
    for ln, was_open, text in odd_lines:
        print(f"        line {ln:5d}  {'CLOSES' if was_open else 'OPENS '}  {text}")

# 5. cross-references. Labels come both from main.tex and from generated table files,
#    where \resulttable{ID} pulls in a file defining \label{tab:ID}.
labels = set(re.findall(r"\\label\{([^}]+)\}", src))
labels |= {f"tab:{t}" for t in re.findall(r"\\resulttable\{([A-Za-z0-9_]+)\}", src)}
refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", src))
report("all refs resolve", sorted(refs - labels))

# 6. citations against the embedded bibliography
keys = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", src))
cited = set()
for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", src):
    cited |= {k.strip() for k in m.group(1).split(",")}
report("all citations defined", sorted(cited - keys))
if keys - cited:
    print(f"  note  {len(keys - cited)} bibitems never cited: {sorted(keys - cited)[:6]}")

# 7. generated-macro fallbacks
used = set(re.findall(r"\\(kw[A-Za-z]+)", src))
declared = set(re.findall(r"\\providecommand\{\\(kw[A-Za-z]+)\}", src))
report("kw macros have fallbacks", sorted(used - declared))

# 8. sections with no body. A heading followed immediately by another heading renders
#    as a numbered line with nothing under it, which is easy to introduce while moving
#    text around and invisible until someone reads the PDF.
#    A \section opening straight into its first \subsection is normal structure, so only
#    a heading followed by one at the same or shallower level counts as empty.
heads = [(m.start(), 2 if m.group(1) else 1, m.group(2))
         for m in re.finditer(r"\\(sub)?section\{([^}]+)\}", src)]
empty = []
for (pos, depth, name), (nxt, ndepth, _) in zip(heads, heads[1:]):
    if ndepth > depth:
        continue
    body = src[pos:nxt]
    body = re.sub(r"\\(sub)?section\{[^}]+\}", "", body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    body = re.sub(r"%.*", "", body)
    if not body.strip():
        empty.append(name)
report("no empty sections", empty)

# 9. artifacts generated but never included. Two tables and two figures had been
#    written to disk by the report script and referenced from nowhere in the manuscript,
#    which is invisible in a compiled PDF: the result simply is not there to miss.
# Only the results paper inputs generated tables; a supplement legitimately has
# none, and flagging every artifact as missing there is noise, not a finding.
report_script = Path(path).resolve().parent.parent / "experiments" / "03_make_paper.py"
if report_script.exists() and re.search(r"\\resulttable\{", src):
    rep = report_script.read_text(encoding="utf-8")
    emitted_tables = set(re.findall(r"write_table\(\s*[\"']([A-Za-z0-9_]+)[\"']", rep))
    used_tables = set(re.findall(r"\\resulttable\{([A-Za-z0-9_]+)\}", src))
    # A table may legitimately live in the companion document instead, so count that
    # as included: what matters is that no generated table goes unpublished.
    companion = Path(path).resolve().parent / "supplementary.tex"
    if companion.exists():
        comp = companion.read_text(encoding="utf-8")
        used_tables |= set(re.findall(r"tables/([A-Za-z0-9_]+)\.tex", comp))
    report("every generated table is included", sorted(emitted_tables - used_tables))

    emitted_figs = set(re.findall(r'FIGURES / ["\']([A-Za-z0-9_]+)\.pdf["\']', rep))
    used_figs = set(re.findall(r"figures/([A-Za-z0-9_]+)\.pdf", src))
    report("every generated figure is included", sorted(emitted_figs - used_figs))
elif not report_script.exists():
    print("  skip  generated-artifact check: report script not alongside")
else:
    print("  skip  generated-artifact check: not the results paper")

# 10. draft-mode state
# Strip comments first: the preamble documents both switch settings in prose, and
# reading the first match found the documentation rather than the switch.
uncommented = re.sub(r"(?m)(?<!\\)%.*$", "", src)
banners = len(re.findall(r"\\pending\{", uncommented))
mode = re.search(r"^\\resultsready(true|false)", uncommented, re.M)
state = mode.group(1) if mode else "unset"
if state == "true":
    note = f"{banners} banner(s) present but inert" if banners else "no banners"
else:
    note = f"{banners} banner(s) will render"
print(f"  info  results mode: \\resultsready{state}   ({note})")
tabs = re.findall(r"\\resulttable\{([A-Za-z0-9_]+)\}", src)
print(f"  info  result tables: {len(tabs)} {sorted(set(tabs))}")

print(f"\n{'PASS' if problems == 0 else str(problems) + ' PROBLEM(S)'}")
sys.exit(1 if problems else 0)
