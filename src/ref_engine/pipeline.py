"""
REF Agentic Pipeline Runner
----------------------------
Chains score → differential → report → export in a single autonomous pass.
Suitable for cron scheduling, n8n webhook triggers, or Oracle Cloud free tier.

Design pattern: each stage produces an artifact; failures are isolated
and reported without halting the pipeline.

CLI:
    python pipeline.py run DIR [--out OUT_DIR] [--pattern GLOB]
    python pipeline.py watch DIR [--interval SECONDS]  # poll for new files
    python pipeline.py serve                            # HTTP endpoint mode
"""

import sys
import os
import json
import time
import glob
import hashlib
import datetime
import argparse
from rich.console import Console

from ref_engine.file_parser import FileParser
from ref_engine.scoring_engine import REFScoringEngine
from ref_engine.report_renderer import REFReportRenderer
from ref_engine.differential_analyser import DifferentialAnalyser
from ref_engine.json_exporter import export_json, score_to_dict

console = Console()


# ─── Pipeline stages ──────────────────────────────────────────────────────────

class REFPipeline:
    """
    Agentic pipeline: parse → score → analyse → export.
    Each stage is fault-isolated. A failed parse does not abort scoring
    of other documents. A failed export does not abort the report.
    """

    def __init__(self, out_dir: str = "/mnt/user-data/outputs",
                 verbose: bool = False):
        self.out_dir = out_dir
        self.verbose = verbose
        self.parser   = FileParser()
        self.engine   = REFScoringEngine()
        self.renderer = REFReportRenderer(console)
        self.analyser = DifferentialAnalyser()
        os.makedirs(out_dir, exist_ok=True)

    def run(self, filepaths: list) -> dict:
        """
        Full pipeline over a list of file paths.
        Returns a summary dict for programmatic use.
        """
        run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        console.print()
        console.rule(f"REF AGENTIC PIPELINE  run={run_id}", style="bold cyan")
        console.print(f"  Files: {len(filepaths)}  →  output: {self.out_dir}\n")

        # ── Stage 1: Parse ────────────────────────────────────────────────────
        console.rule("STAGE 1 · PARSE", style="dim")
        docs = []
        parse_failures = []
        for fp in filepaths:
            try:
                doc = self.parser.parse(fp)
                if doc.word_count < 30:
                    console.print(f"  [yellow]⚠  {os.path.basename(fp)}: "
                                  f"low word count ({doc.word_count}) — skipping[/]")
                    parse_failures.append(fp)
                else:
                    docs.append(doc)
                    console.print(f"  [green]✓[/]  {os.path.basename(fp):40}  "
                                  f"{doc.word_count:,} words  "
                                  f"{doc.page_count} pages  "
                                  f"{doc.equation_count} eqs")
            except Exception as e:
                parse_failures.append(fp)
                console.print(f"  [red]✗  {os.path.basename(fp)}: {e}[/]")

        console.print()

        if not docs:
            console.print("[red]No documents successfully parsed. Aborting.[/]")
            return {"status": "aborted", "reason": "no_parsed_docs"}

        # ── Stage 2: Score ────────────────────────────────────────────────────
        console.rule("STAGE 2 · SCORE", style="dim")
        scores = []
        score_failures = []
        for doc in docs:
            try:
                t0 = time.perf_counter()
                score = self.engine.score(doc)
                elapsed = time.perf_counter() - t0
                scores.append(score)

                vcolors = {
                    "RECOGNITIVE": "bright_green",
                    "MANIPULATIVE": "bright_red",
                    "HIGH_DISTORTION": "orange1",
                }
                vc = vcolors.get(score.verdict, "yellow")
                console.print(
                    f"  [{vc}]{score.verdict:<22}[/]  "
                    f"{score.overall_score():.4f}  "
                    f"{doc.filename}  "
                    f"[dim]({elapsed:.2f}s)[/]"
                )
            except Exception as e:
                score_failures.append(doc.filename)
                console.print(f"  [red]✗  {doc.filename}: {e}[/]")

        console.print()

        # ── Stage 3: Individual reports ───────────────────────────────────────
        console.rule("STAGE 3 · INDIVIDUAL REPORTS", style="dim")
        report_paths = []
        for score in scores:
            try:
                out_path = os.path.join(
                    self.out_dir,
                    f"ref_{_sanitize(score.document_name)}_{run_id}.json"
                )
                export_json(score, out_path)
                report_paths.append(out_path)
                console.print(f"  [dim green]✓[/]  {os.path.basename(out_path)}")
                if self.verbose:
                    self.renderer.render(score, verbose=True)
            except Exception as e:
                console.print(f"  [red]✗  {score.document_name}: {e}[/]")

        console.print()

        # ── Stage 4: Differential analysis ───────────────────────────────────
        console.rule("STAGE 4 · DIFFERENTIAL ANALYSIS", style="dim")
        corpus_analysis = None
        if len(scores) >= 2:
            try:
                corpus_analysis = self.analyser.analyse(scores)
                self.analyser.render_analysis(corpus_analysis, console)

                # Export corpus analysis
                corpus_out = os.path.join(
                    self.out_dir, f"ref_corpus_analysis_{run_id}.json"
                )
                corpus_dict = {
                    "run_id": run_id,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "documents": [s.document_name for s in scores],
                    "axis_variances": corpus_analysis.axis_variances,
                    "top_discriminant": corpus_analysis.top_discriminant,
                    "centroid": corpus_analysis.centroid,
                    "outliers": corpus_analysis.outliers,
                    "ranking": [
                        {"rank": i+1, "document": p.name,
                         "overall": round(p.overall, 4), "verdict": p.verdict}
                        for i, p in enumerate(
                            sorted(corpus_analysis.profiles,
                                   key=lambda x: x.overall, reverse=True)
                        )
                    ],
                    "pairwise_contrasts": [
                        {"doc_a": c.doc_a, "doc_b": c.doc_b,
                         "euclidean": round(c.euclidean_distance, 4),
                         "cosine": round(c.cosine_similarity, 4),
                         "summary": c.summary}
                        for c in corpus_analysis.pairwise
                    ],
                    "per_document_scores": [score_to_dict(s) for s in scores],
                }
                with open(corpus_out, "w") as f:
                    json.dump(corpus_dict, f, indent=2, ensure_ascii=False)
                console.print(f"  [dim green]✓  Corpus analysis → {corpus_out}[/]")

            except Exception as e:
                console.print(f"  [red]✗  Differential analysis failed: {e}[/]")
        else:
            console.print("  [dim](need ≥2 documents for differential analysis)[/]")

        console.print()

        # ── Stage 5: Summary manifest ─────────────────────────────────────────
        console.rule("STAGE 5 · MANIFEST", style="dim")
        manifest = {
            "run_id":           run_id,
            "timestamp":        datetime.datetime.utcnow().isoformat() + "Z",
            "files_input":      len(filepaths),
            "parse_failures":   parse_failures,
            "score_failures":   score_failures,
            "documents_scored": len(scores),
            "report_files":     report_paths,
            "verdicts": {
                s.document_name: {
                    "verdict": s.verdict,
                    "overall": round(s.overall_score(), 4),
                    "recognition":  round(s.mean_recognition,  4),
                    "fidelity_dist":round(s.mean_fidelity,      4),
                    "hierarchy":    round(s.mean_hierarchy,     4),
                    "concision":    round(s.mean_concision,     4),
                }
                for s in scores
            },
            "ranking": sorted(
                [{"document": s.document_name, "overall": round(s.overall_score(),4),
                  "verdict": s.verdict} for s in scores],
                key=lambda x: x["overall"], reverse=True
            ),
        }

        manifest_path = os.path.join(self.out_dir,
                                     f"ref_manifest_{run_id}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        console.print(f"  [bold green]✓  Manifest → {manifest_path}[/]")
        console.print()
        console.rule("PIPELINE COMPLETE", style="bold cyan")
        console.print(
            f"\n  {len(scores)} documents scored  ·  "
            f"{len(parse_failures)} parse failures  ·  "
            f"run_id={run_id}\n"
        )

        return manifest


class FileWatcher:
    """
    Poll a directory for new files and trigger the pipeline automatically.
    Tracks processed files via SHA256 hash to avoid re-processing.
    """

    def __init__(self, dirpath: str, pipeline: REFPipeline,
                 interval: float = 30.0, pattern: str = "*.pdf"):
        self.dirpath  = dirpath
        self.pipeline = pipeline
        self.interval = interval
        self.pattern  = pattern
        self.seen:    set = set()

    def watch(self) -> None:
        console.print(f"\n  [cyan]REF FileWatcher[/] polling "
                      f"{self.dirpath} every {self.interval}s\n")
        try:
            while True:
                new_files = self._find_new()
                if new_files:
                    console.print(f"  [green]→ {len(new_files)} new files detected[/]")
                    self.pipeline.run(new_files)
                    for fp in new_files:
                        self.seen.add(self._hash(fp))
                time.sleep(self.interval)
        except KeyboardInterrupt:
            console.print("\n  [dim]FileWatcher stopped.[/]")

    def _find_new(self) -> list:
        files = glob.glob(os.path.join(self.dirpath, self.pattern))
        new = []
        for fp in files:
            h = self._hash(fp)
            if h not in self.seen:
                new.append(fp)
        return new

    def _hash(self, filepath: str) -> str:
        try:
            stat = os.stat(filepath)
            key  = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.sha256(key.encode()).hexdigest()[:16]
        except OSError:
            return filepath


# ─── Utilities ────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """Make a filename-safe version of a document name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="pipeline",
        description="REF Agentic Pipeline Runner",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline over files/directory")
    p_run.add_argument("path", help="File, files, or directory")
    p_run.add_argument("--out", default="/mnt/user-data/outputs",
                       help="Output directory")
    p_run.add_argument("--pattern", default="*.pdf",
                       help="Glob pattern when path is a directory")
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_watch = sub.add_parser("watch", help="Watch directory for new files")
    p_watch.add_argument("directory", help="Directory to watch")
    p_watch.add_argument("--interval", type=float, default=30.0,
                         help="Polling interval in seconds")
    p_watch.add_argument("--pattern", default="*.pdf")
    p_watch.add_argument("--out", default="/mnt/user-data/outputs")

    args = ap.parse_args()

    if args.command == "run":
        pipeline = REFPipeline(out_dir=args.out, verbose=args.verbose)
        if os.path.isdir(args.path):
            filepaths = sorted(glob.glob(
                os.path.join(args.path, args.pattern)
            ))
        elif os.path.isfile(args.path):
            filepaths = [args.path]
        else:
            # treat as glob
            filepaths = sorted(glob.glob(args.path))

        if not filepaths:
            console.print(f"[red]No files found at: {args.path}[/]")
            sys.exit(1)

        pipeline.run(filepaths)

    elif args.command == "watch":
        pipeline = REFPipeline(out_dir=args.out)
        watcher  = FileWatcher(
            dirpath=args.directory,
            pipeline=pipeline,
            interval=args.interval,
            pattern=args.pattern,
        )
        watcher.watch()


if __name__ == "__main__":
    main()
