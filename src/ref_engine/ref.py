#!/usr/bin/env python3
"""
REF — Recognitive Equation Framework
======================================
Agentic CLI for scoring documents against the REF validity criterion:

    F is RECOGNITIVE iff:
        Rec(F,Ψ₀) > θ_r = 0.45     (recognition)
        ℰ_F       < θ_f = 0.55     (low distortion)
        ℋ(T)      < θ_h = 0.40     (low hierarchy)
        C(T)      > θ_c = 0.30     (concision)

Usage:
    python ref.py score FILE [FILE ...]
    python ref.py score FILE --verbose --json out.json
    python ref.py batch DIR --out results/
    python ref.py compare FILE1 FILE2 [FILE3 ...]
    python ref.py reset TEXT         # apply Pause→Notice→Return to a text
    python ref.py demo               # run against all uploaded PDFs

Author: REF Engine v1.0
"""

import sys
import os
import json
import time
import glob
import argparse
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from ref_engine.quaternion import QuaternionState
from ref_engine.feature_extractor import TextFeatureExtractor
from ref_engine.file_parser import FileParser
from ref_engine.scoring_engine import REFScoringEngine, RecognitiveScore
from ref_engine.report_renderer import REFReportRenderer
from ref_engine.json_exporter import export_json, export_jsonl, score_to_dict


console = Console()


# ─── Core pipeline ────────────────────────────────────────────────────────────

def score_file(filepath: str, verbose: bool = False,
               json_out: str = None) -> RecognitiveScore:
    """
    Full pipeline: parse → extract → score → render.
    Returns the RecognitiveScore for downstream use.
    """
    parser  = FileParser()
    engine  = REFScoringEngine(verbose=verbose)
    renderer = REFReportRenderer(console)

    console.print(f"\n  [dim cyan]⟳  Parsing:[/]  {filepath}")
    t0 = time.perf_counter()

    doc   = parser.parse(filepath)
    score = engine.score(doc)

    elapsed = time.perf_counter() - t0
    console.print(f"  [dim]   scored {len(score.chunk_scores)} chunks "
                  f"in {elapsed:.2f}s[/]")

    renderer.render(score, verbose=verbose)

    if json_out:
        export_json(score, json_out)
        console.print(f"  [dim green]✓  JSON exported → {json_out}[/]")

    return score


def compare_files(filepaths: list, json_out: str = None) -> None:
    """
    Score multiple files and render a side-by-side comparison table.
    This is the most powerful single operation in the REF toolkit —
    it turns the abstract framework into a discrimination instrument.
    """
    parser  = FileParser()
    engine  = REFScoringEngine()

    scores = []
    for fp in filepaths:
        console.print(f"  [dim cyan]⟳  Scoring:[/]  {fp}")
        doc   = parser.parse(fp)
        sc    = engine.score(doc)
        scores.append(sc)

    if not scores:
        console.print("[red]No files scored.[/]")
        return

    console.rule("COMPARATIVE ANALYSIS", style="bold cyan")
    console.print()

    # Comparison table
    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, padding=(0, 1),
                title="[bold cyan]REF Comparative Score Matrix[/]")
    tbl.add_column("Document",    style="cyan",  width=32)
    tbl.add_column("Verdict",     style="white", width=22)
    tbl.add_column("Overall",     width=9)
    tbl.add_column("Rec",         width=7)
    tbl.add_column("1−ℰ",         width=7)
    tbl.add_column("1−ℋ",         width=7)
    tbl.add_column("C",           width=7)
    tbl.add_column("Grnd",        width=7)
    tbl.add_column("Affect",      width=7)
    tbl.add_column("τ",           width=8)

    VERDICT_COLORS = {
        "RECOGNITIVE":           "bright_green",
        "PARTIALLY_RECOGNITIVE": "yellow",
        "NEUTRAL":               "cyan",
        "HIGH_DISTORTION":       "orange1",
        "MANIPULATIVE":          "bright_red",
        "UNSCORED":              "dim",
        "EMPTY_DOCUMENT":        "dim",
    }

    def cell(v, invert=False):
        pct = (1-v) if invert else v
        color = "bright_green" if pct >= .65 else "yellow" if pct >= .45 else "red"
        return Text(f"{v:.3f}", style=f"bold {color}")

    for s in scores:
        vcolor = VERDICT_COLORS.get(s.verdict, "white")
        tbl.add_row(
            s.document_name[:30],
            Text(s.verdict, style=f"bold {vcolor}"),
            cell(s.overall_score()),
            cell(s.mean_recognition),
            cell(1 - s.mean_fidelity),
            cell(1 - s.mean_hierarchy),
            cell(s.mean_concision),
            cell(s.mean_grounding),
            cell(s.mean_affect),
            Text(f"{s.mean_temporal:+.3f}", style="dim"),
        )

    console.print(tbl)
    console.print()

    # Ranking
    console.rule("RANKING  (by overall recognitive score)", style="dim")
    console.print()
    ranked = sorted(scores, key=lambda s: s.overall_score(), reverse=True)
    for i, s in enumerate(ranked, 1):
        vcolor = VERDICT_COLORS.get(s.verdict, "white")
        console.print(
            f"  [{i}]  {s.overall_score():.4f}  "
            f"[{vcolor}]{s.verdict:<25}[/]  {s.document_name}",
        )
    console.print()

    # Discriminant insight: which axis best separates the documents?
    if len(scores) >= 2:
        import numpy as np
        axes = ["mean_recognition", "mean_fidelity", "mean_hierarchy",
                "mean_concision", "mean_grounding", "mean_affect"]
        variances = {}
        for ax in axes:
            vals = [getattr(s, ax) for s in scores]
            variances[ax] = float(np.var(vals))
        top_discriminant = max(variances, key=variances.get)
        console.print(
            f"  [dim]Most discriminating axis: "
            f"[bold]{top_discriminant}[/]  "
            f"(variance={variances[top_discriminant]:.4f})[/]"
        )
        console.print()

    if json_out:
        all_data = [score_to_dict(s) for s in scores]
        with open(json_out, "w") as f:
            json.dump(all_data, f, indent=2)
        console.print(f"  [dim green]✓  JSON exported → {json_out}[/]")


def demo_reset(text: str) -> None:
    """
    Apply the Pause→Notice→Return cycle to a text snippet.
    Demonstrates the minimal recognitive reset law:
        Ψ_{k+1} = Π₀(𝒩(𝒫(Ψ_k)))
    """
    extractor = TextFeatureExtractor()
    feat = extractor.extract(text)

    # Construct initial state from text features
    psi = QuaternionState.from_features(
        attention = feat.attention_stability,
        grounding = feat.sensory_grounding,
        memory    = feat.memory_depth,
        affect    = feat.affective_valence,
    )

    console.rule("RECOGNITIVE RESET  Pause→Notice→Return", style="cyan")
    console.print(f"\n  [dim]Input text:[/] {text[:120]}…\n")
    console.print("  [cyan]Extracted features:[/]")
    console.print(f"    synthetic pressure = {feat.synthetic_pressure:.4f}")
    console.print(f"    hierarchy          = {feat.hierarchy_score:.4f}")
    console.print(f"    concision          = {feat.concision_density:.4f}")
    console.print()
    console.print(f"  [bold]Ψ_initial:[/]  {psi}")

    from ref_engine.quaternion import pause_operator, notice_operator, return_operator
    psi_paused  = pause_operator(psi, feat.synthetic_pressure)
    psi_noticed = notice_operator(psi_paused)
    psi_returned = return_operator(psi_noticed)

    console.print(f"  [yellow]Ψ_paused: [/]  {psi_paused}")
    console.print(f"  [cyan]Ψ_noticed:[/]  {psi_noticed}")
    console.print(f"  [green]Ψ_returned:[/] {psi_returned}")
    console.print()

    source = QuaternionState.baseline()
    fidelity_before = source.fidelity(psi)
    fidelity_after  = source.fidelity(psi_returned)
    console.print(
        f"  Fidelity to S₀:  "
        f"before={fidelity_before:.4f}  →  "
        f"after={fidelity_after:.4f}  "
        f"(Δ={fidelity_after - fidelity_before:+.4f})"
    )
    console.print()


def analyse_deep(filepaths: list, json_out: str = None) -> None:
    """Deep analysis: corpus index + diff + trajectory + decomposition + signals."""
    from ref_engine.analytics import (DiffEngine, TemporalEvolution, SignalExtractor,
                            CrossCorpusIndex, OperatorDecomposer)
    from ref_engine.analytics_renderer import AnalyticsRenderer
    ar = AnalyticsRenderer(console)
    _parser = FileParser()
    _engine = REFScoringEngine()
    scores = []
    for fp in filepaths:
        console.print(f"  [dim cyan]⟳  Scoring:[/]  {fp}")
        doc = _parser.parse(fp)
        s   = _engine.score(doc)
        scores.append(s)
    if not scores:
        return
    idx = CrossCorpusIndex()
    for s in scores:
        idx.add(s)
    ar.render_corpus_index(idx)
    if len(scores) >= 2:
        diff = DiffEngine().diff(scores[0], scores[1])
        ar.render_diff(diff)
    evolver    = TemporalEvolution()
    decomposer = OperatorDecomposer()
    extractor  = SignalExtractor()
    for s in scores:
        if s.chunk_scores:
            ar.render_trajectory(evolver.extract(s), axis="recognition")
        ar.render_decomposition(decomposer.decompose(s))
        if s.chunk_scores:
            console.print(f"\n  [cyan]{s.document_name}[/]")
            ar.render_signals(extractor.extract(s))
    if json_out:
        output = {
            "corpus_index": idx.to_dict(),
            "scores": [score_to_dict(s) for s in scores],
            "trajectories": [evolver.extract(s).to_dict()
                             for s in scores if s.chunk_scores],
        }
        with open(json_out, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\n  [green]✓  Deep analysis JSON → {json_out}[/]")


def batch_directory(dirpath: str, out_dir: str = None,
                    pattern: str = "*.pdf") -> None:
    """Score all matching files in a directory."""
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not files:
        console.print(f"[red]No {pattern} files found in {dirpath}[/]")
        return

    console.print(f"\n  [cyan]Batch scoring {len(files)} files...[/]\n")
    scores = []
    parser = FileParser()
    engine = REFScoringEngine()

    for fp in files:
        console.print(f"  [dim]  → {os.path.basename(fp)}[/]", end=" ")
        doc = parser.parse(fp)
        s   = engine.score(doc)
        scores.append(s)
        vcolor = {"RECOGNITIVE": "green", "MANIPULATIVE": "red"}.get(
            s.verdict, "yellow")
        console.print(f"[{vcolor}]{s.verdict}[/]  {s.overall_score():.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "batch_results.jsonl")
        export_jsonl(scores, out_path)
        console.print(f"\n  [green]✓ JSONL results saved → {out_path}[/]")


# ─── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ref",
        description="REF — Recognitive Equation Framework scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # score
    p_score = sub.add_parser("score", help="Score one or more files")
    p_score.add_argument("files", nargs="+", help="File paths to score")
    p_score.add_argument("-v", "--verbose", action="store_true",
                         help="Show chunk-level detail")
    p_score.add_argument("-j", "--json",    metavar="PATH",
                         help="Export JSON report to this path")

    # compare
    p_cmp = sub.add_parser("compare",
                           help="Side-by-side comparison of multiple files")
    p_cmp.add_argument("files", nargs="+", help="Files to compare")
    p_cmp.add_argument("-j", "--json", metavar="PATH",
                       help="Export comparison JSON")

    # batch
    p_batch = sub.add_parser("batch", help="Score all PDFs in a directory")
    p_batch.add_argument("directory", help="Directory to scan")
    p_batch.add_argument("--pattern", default="*.pdf",
                         help="Glob pattern (default: *.pdf)")
    p_batch.add_argument("--out", metavar="DIR",
                         help="Output directory for JSONL results")

    # reset
    p_reset = sub.add_parser("reset",
                              help="Apply Pause→Notice→Return to a text snippet")
    p_reset.add_argument("text", help="Text string to process")

    # analyse (deep)
    p_an = sub.add_parser("analyse",
                          help="Deep analysis: diff + trajectory + decomposition + signals")
    p_an.add_argument("files", nargs="+", help="Files to analyse (2+ recommended)")
    p_an.add_argument("-j", "--json", metavar="PATH",
                      help="Export full deep-analysis JSON")

    # demo
    sub.add_parser("demo", help="Score all uploaded PDFs and compare them")

    return ap


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.command == "score":
        for fp in args.files:
            score_file(fp, verbose=args.verbose, json_out=args.json)

    elif args.command == "compare":
        compare_files(args.files, json_out=args.json)

    elif args.command == "batch":
        batch_directory(args.directory, out_dir=args.out,
                        pattern=args.pattern)

    elif args.command == "reset":
        demo_reset(args.text)

    elif args.command == "analyse":
        analyse_deep(args.files, json_out=args.json)

    elif args.command == "demo":
        upload_dir = "/mnt/user-data/uploads"
        pdfs = sorted(glob.glob(os.path.join(upload_dir, "*.pdf")))
        if not pdfs:
            console.print("[red]No PDFs found in upload directory.[/]")
            sys.exit(1)
        console.print(f"\n  [bold cyan]REF Demo — scoring {len(pdfs)} uploaded documents[/]\n")
        compare_files(pdfs, json_out="/mnt/user-data/outputs/ref_demo_comparison.json")


if __name__ == "__main__":
    main()
