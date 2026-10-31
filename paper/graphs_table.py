"""tables/graphs.tex from results/p13_timing_graph.json: released sampler vs RT6 on the TinyStories model, one RTX 4090,
eager vs torch.compile (fusion) vs torch.compile reduce-overhead (fusion + CUDA-graph replay), batch 128 and 8."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "p13_timing_graph.json")
OUT = os.path.join(HERE, "tables", "graphs.tex")
NAMES = {"eager": "eager (as released)", "compile": "compiled, fusion only", "graphs": "compiled + CUDA graphs"}

d = json.load(open(SRC))
rows = {(r["mode"], r["batch"], r["sampler"]): r for r in d["rows"] if "error" not in r}
lines = []
for B in (128, 8):
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{batch {B}}}}} \\\\")
    base = rows[("eager", B, "released")]["ms_per_sample"]
    for mode in ("eager", "compile", "graphs"):
        rel, rt = rows[(mode, B, "released")], rows[(mode, B, f"RT{d['k']}")]
        a, b = rel["ms_per_sample"], rt["ms_per_sample"]
        lines.append(f"{NAMES[mode]} & {a:.2f} & {b:.2f} & {b / a:.2f}$\\times$ & {a / base:.2f}$\\times$ & {b / base:.2f}$\\times$ \\\\")
    if B == 128:
        lines.append("\\midrule")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
for ln in lines:
    print(ln)
