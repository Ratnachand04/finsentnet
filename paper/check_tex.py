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

# 9. draft-mode state
banners = len(re.findall(r"\\pending\{", src))
mode = re.search(r"\\resultsready(true|false)", src)
print(f"  info  pending banners: {banners}   results mode: "
      f"{mode.group(0) if mode else 'unset'}")
tabs = re.findall(r"\\resulttable\{([A-Za-z0-9_]+)\}", src)
print(f"  info  result tables: {len(tabs)} {sorted(set(tabs))}")

print(f"\n{'PASS' if problems == 0 else str(problems) + ' PROBLEM(S)'}")
sys.exit(1 if problems else 0)
