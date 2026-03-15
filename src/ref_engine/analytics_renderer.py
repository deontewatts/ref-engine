"""
REF Analytics Renderer
-----------------------
Rich terminal rendering for advanced analytics outputs:
  - DocumentDiff display
  - StateTrajectory ASCII chart
  - OperatorDecomposition attribution view
  - CrossCorpusIndex summary
  - SignalExtractor output
"""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import box

from ref_engine.analytics import (DocumentDiff, StateTrajectory, OperatorDecomposition,
                       CrossCorpusIndex)
from ref_engine.report_renderer import score_color, bar


class AnalyticsRenderer:

    def __init__(self, console: Console = None):
        self.console = console or Console(highlight=False)

    # ── Diff view ─────────────────────────────────────────────────────────────

    def render_diff(self, diff: DocumentDiff) -> None:
        c = self.console
        c.rule("DOCUMENT DIFF", style="bold cyan")
        c.print()

        c.print(f"  [cyan]A[/]  {diff.doc_a}")
        c.print(f"  [cyan]B[/]  {diff.doc_b}")
        c.print()

        tbl = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
        tbl.add_column("Axis",      style="cyan",  width=18)
        tbl.add_column("Δ (B−A)",   width=10)
        tbl.add_column("Direction", width=32)

        AXIS_ROWS = [
            ("Recognition",    diff.delta_recognition,  False),
            ("1−Fidelity_Δ",   diff.delta_fidelity,     False),
            ("1−Hierarchy",    -diff.delta_hierarchy,   False),  # invert: less hier = better
            ("Concision",      diff.delta_concision,    False),
            ("Grounding",      diff.delta_grounding,    False),
            ("Affect",         diff.delta_affect,       False),
            ("Temporal τ",     diff.delta_temporal,     False),
            ("OVERALL",        diff.delta_overall,      False),
        ]

        for name, delta, _inv in AXIS_ROWS:
            if abs(delta) < 1e-6:
                color, arrow = "dim", "≈ equal"
            elif delta > 0:
                color, arrow = "bright_green", f"▲ B better by {abs(delta):.4f}"
            else:
                color, arrow = "red",          f"▼ A better by {abs(delta):.4f}"
            weight = "bold " if name == "OVERALL" else ""
            tbl.add_row(
                Text(name, style=f"{weight}{'yellow' if name == 'OVERALL' else 'cyan'}"),
                Text(f"{delta:+.4f}", style=f"{weight}{color}"),
                Text(arrow, style=color),
            )  # noqa: E225

        c.print(tbl)
        c.print()
        c.print(f"  Quaternion divergence ∥ΔΨ∥ = {diff.psi_divergence:.4f}",
                style="dim")
        c.print(f"  Dominant axis:  [bold]{diff.dominant_difference}[/]", style="dim")
        c.print()
        c.print(Panel(diff.interpretation, border_style="dim cyan",
                      title="[dim]Interpretation[/]"))
        c.print()

    # ── Trajectory ASCII chart ─────────────────────────────────────────────────

    def render_trajectory(self, traj: StateTrajectory,
                          axis: str = "recognition") -> None:
        c = self.console
        c.rule(f"STATE TRAJECTORY  [{axis}]  ·  {traj.document_name}",
               style="bold cyan")
        c.print()

        trace_map = {
            "recognition": traj.recognition_trace,
            "hierarchy":   traj.hierarchy_trace,
            "fidelity":    traj.fidelity_trace,
            "psi0":        traj.psi0_trace,
            "psi3":        traj.psi3_trace,
        }
        trace = trace_map.get(axis, traj.recognition_trace)

        if not trace:
            c.print("  [dim]No trajectory data.[/]")
            return

        HEIGHT = 10
        min_v  = min(trace) - 0.05
        max_v  = max(trace) + 0.05
        span   = max_v - min_v or 1.0

        # Build ASCII chart rows top-to-bottom
        chart_lines = []
        for row in range(HEIGHT, -1, -1):
            threshold = min_v + (row / HEIGHT) * span
            y_label = f"{threshold:5.2f} │"
            row_str = ""
            for val in trace:
                if val >= threshold - (span / HEIGHT / 2):
                    row_str += " ●"
                else:
                    row_str += "  "
            chart_lines.append(f"  {y_label}{row_str}")

        # X-axis
        x_axis = "       └" + "──" * len(trace) + "▶ chunks"
        chunk_labels = "         " + "".join(
            f"{i:<2}" for i in traj.chunk_indices
        )

        for line in chart_lines:
            c.print(line, style="cyan")
        c.print(x_axis, style="dim")
        c.print(chunk_labels, style="dim")
        c.print()

        # Stats
        v_arr = traj.recognition_trace if axis == "recognition" else trace
        import numpy as np
        c.print(
            f"  mean={np.mean(v_arr):.3f}  "
            f"std={np.std(v_arr):.3f}  "
            f"volatility={traj.volatility(axis):.4f}  "
            f"turning_points={traj.turning_points(axis)}",
            style="dim"
        )
        c.print()

    # ── Operator decomposition view ────────────────────────────────────────────

    def render_decomposition(self, decomp: OperatorDecomposition) -> None:
        c = self.console
        c.rule("OPERATOR DECOMPOSITION", style="bold cyan")
        c.print()
        c.print(f"  [cyan]{decomp.document_name}[/]")
        c.print()

        tbl = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
        tbl.add_column("Operator",    style="cyan", width=16)
        tbl.add_column("Weight",      width=8)
        tbl.add_column("Contribution",width=14)
        tbl.add_column("Attribution bar", width=30)

        total = sum(decomp.operator_contributions.values())
        for op, contrib in sorted(decomp.operator_contributions.items(),
                                   key=lambda x: x[1], reverse=True):
            weight = {
                "Recognition": 0.25, "Fidelity": 0.30, "Egalitarian": 0.25,
                "Concision":   0.10, "Grounding": 0.10,
            }.get(op, 0.0)
            pct = contrib / (total + 1e-9)
            color = score_color(pct)
            is_top    = "  ◀ TOP" if op == decomp.top_driver    else ""
            is_bottom = "  ◀ GAP" if op == decomp.bottom_driver else ""
            tbl.add_row(
                op,
                f"{weight:.2f}",
                Text(f"{contrib:.4f}{is_top}{is_bottom}",
                     style=f"bold {color}" if is_bottom else color),
                Text(bar(pct), style=color),
            )

        c.print(tbl)
        c.print()
        c.print(f"  [bold]Primary bottleneck:[/]  {decomp.bottom_driver}")
        c.print("  [bold]Prescription:[/]")
        c.print(f"    {decomp.prescription}", style="dim")
        c.print()

        if decomp.bottleneck_passages:
            c.print("  [yellow]Bottleneck passages (lowest on limiting axis):[/]")
            for i, p in enumerate(decomp.bottleneck_passages, 1):
                c.print(f"  [{i}] {p}", style="dim")
            c.print()

        if decomp.enhancement_passages:
            c.print("  [green]Enhancement passages (strongest signals):[/]")
            for i, p in enumerate(decomp.enhancement_passages, 1):
                c.print(f"  [{i}] {p}", style="dim")
            c.print()

    # ── Corpus index view ──────────────────────────────────────────────────────

    def render_corpus_index(self, index: CrossCorpusIndex) -> None:
        c = self.console
        c.rule("CROSS-CORPUS INDEX", style="bold cyan")
        c.print()

        data = index.to_dict()
        stats = data.get("statistics", {})
        clusters = data.get("clusters", {})

        # Stats table
        c.print(f"  Corpus size: [bold]{data['corpus_size']}[/] documents\n")

        if stats:
            st = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
            st.add_column("Axis",   style="cyan", width=18)
            st.add_column("Mean",   width=8)
            st.add_column("Std",    width=8)
            st.add_column("Range",  width=16)
            for axis, s in stats.items():
                st.add_row(
                    axis,
                    f"{s['mean']:.3f}",
                    f"{s['std']:.3f}",
                    f"[{s['min']:.3f}, {s['max']:.3f}]",
                )
            c.print(st)
            c.print()

        # Clusters
        c.print("  [bold]Verdict clusters:[/]")
        for verdict, docs in sorted(clusters.items()):
            color = {"RECOGNITIVE": "green", "MANIPULATIVE": "red",
                     "NEUTRAL": "cyan"}.get(verdict, "yellow")
            c.print(f"    [{color}]{verdict}[/]  ({len(docs)})")
            for d in docs:
                c.print(f"      · {d}", style="dim")
        c.print()

    # ── Signal extractor view ──────────────────────────────────────────────────

    def render_signals(self,
                       signals: dict,
                       max_per_type: int = 3) -> None:
        c = self.console
        c.rule("EXTRACTED SIGNALS", style="bold cyan")
        c.print()

        TYPE_COLORS = {
            "HIGH_RECOGNITION": "bright_green",
            "HIGH_DISTORTION":  "bright_red",
            "HIERARCHY_PEAK":   "yellow",
        }

        for sig_type, signal_list in signals.items():
            color = TYPE_COLORS.get(sig_type, "white")
            c.print(f"  [{color}]▸ {sig_type}[/]")
            for sig in signal_list[:max_per_type]:
                c.print(f"    chunk [{sig.chunk_index:03d}]  "
                        f"score={sig.score:.4f}",
                        style=f"dim {color}")
                c.print(f"    {sig.text_preview}", style="dim")
                for k, v in sig.axis_values.items():
                    c.print(f"      {k}={v:.4f}", style="dim")
            c.print()
