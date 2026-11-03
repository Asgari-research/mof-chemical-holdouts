#!/usr/bin/env python3
"""Check immutable V6 figure assets and selected compact numerical invariants.

Run from any directory: python revision_v6/public_release/verify_public_release.py
Does not regenerate or modify figures; no third-party dependencies.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript/v6"
FIGURES = MANUSCRIPT / "figures"
MANIFEST = Path(__file__).with_name("FINAL_FIGURE_SHA256.csv")

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def near(a,b,tol=1e-10):
    require(abs(float(a)-float(b))<tol, f"Expected {b}, found {a}")

def check_figures():
    with MANIFEST.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==22, f"22 figure manifest rows required, found {len(rows)}")
    paths=set()
    for row in rows:
        name=row["filename"]
        path=FIGURES/name
        require(name not in paths, f"Duplicate manifest filename: {name}")
        paths.add(name)
        require(path.is_file(),f"Missing figure: {path}")
        content=path.read_bytes()
        require(content.startswith(b"%PDF-"),f"Not a PDF: {name}")
        require(len(content)==int(row["bytes"]),f"Byte count differs: {name}")
        require(hashlib.sha256(content).hexdigest()==row["sha256"], f"SHA-256 differs: {name}")
    actual={p.name for p in FIGURES.glob("*.pdf")}
    require(actual==paths, f"Unexpected/missing PDF assets: {sorted(actual^paths)}")
    print(f"PASS: {len(rows)} final figure PDFs have their approved SHA-256 checksums")
    return paths

def check_latex(paths):
    pattern=re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    for texname,expected in [("Main_Manuscript_Advanced_Science_V6.tex",9),("Supporting_Information_Advanced_Science_V6.tex",13)]:
        content=(MANUSCRIPT/texname).read_text(encoding="utf-8")
        found=pattern.findall(content)
        require(len(found)==expected,f"Expected {expected} figures in {texname}, got {len(found)}")
        for ref in found:
            if ref.startswith("\\figmain/") or ref.startswith("\\figsi/"):
                fname=ref.split("/",1)[1]
            else:
                raise AssertionError(f"Unrecognized figure path: {ref}")
            require(fname in paths,f"Missing locked figure in {texname}: {fname}")
        print(f"PASS: {texname} has {len(found)} resolvable figure references")

def check_aggregate_arithmetic():
    f=MANUSCRIPT/'v6_source_data/matched_random_vs_topology_fold_metrics.csv'
    with f.open(newline="",encoding="utf-8") as inp:
        rows=list(csv.DictReader(inp))
    require(len(rows)==5,"Expected five matched outer fold records")
    require({int(r["fold"]) for r in rows}==set(range(5)),"Fold IDs 0–4 missing")
    for r in rows:
        require(int(r['train_n'])+int(r['test_n'])==263744, f"Population mismatch in fold {r['fold']}")
    mean=lambda k: sum(float(r[k]) for r in rows)/len(rows)
    near(mean('random_r2'),0.7579863769992089,1e-12)
    near(mean('topology_r2'),0.2674461894481439,1e-12)
    require(all(float(r['random_r2'])>float(r['topology_r2']) for r in rows),"R2 ordering differs")
    near(mean('random_top5_recovery'),0.6219519464463109,1e-12)
    near(mean('topology_top5_recovery'),0.3759120114331019,1e-12)
    print('PASS: matched-control fold sizes, means, and R2 order')
    d=json.loads((MANUSCRIPT/'v6_source_data/pairwise_summary.json').read_text(encoding='utf-8'))
    inter=int(d['both'])
    a=int(d['random_only'])
    b=int(d['topology_only'])
    require(inter+a==int(d['top_k'])==inter+b,'Shortlist cardinalities do not match')
    near(inter/(inter+a+b),float(d['jaccard']),1e-12)
    require(int(d['common_population_n'])==177411,'Common population does not match frozen summary')
    print('PASS: common-universe shortlist counts and Jaccard')

def main():
    try:
        paths=check_figures()
        check_latex(paths)
        check_aggregate_arithmetic()
    except Exception as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        return 1
    print('All public-release integrity tests passed. This is not a fresh model fit.')
    return 0

if __name__=='__main__':
    sys.exit(main())
