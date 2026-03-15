"""
REF Report Renderer
--------------------
Produces a rich terminal report from a RecognitiveScore.
Uses only the 'rich' library (available) for beautiful, structured output.

Design aesthetic: high-signal, low-noise — engineering terminal, not marketing.
Information density is a feature, not a bug.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ref_engine.scoring_engine import RecognitiveScore


# ─── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "RECOGNITIVE":           ("bright_green",  "✦"),
    "PARTIALLY_RECOGNITIVE": ("yellow",        "◈"),
    "NEUTRAL":               ("cyan",          "◇"),
    "HIGH_DISTORTION":       ("red",           "⚠"),
    "MANIPULATIVE":          ("bright_red",    "✗"),
    "UNSCORED":              ("dim white",     "·"),
    "EMPTY_DOCUMENT":        ("dim",           "·"),
}


def score_color(value: float, invert: bool = False) -> str:
    """Map a [0,1] score to a terminal color. invert=True for lower-is-better."""
    if invert:
        value = 1.0 - value
    if value >= 0.70:   return "bright_green"
    if value >= 0.50:   return "yellow"
    if value >= 0.30:   return "orange1"
    return "bright_red"


def bar(value: float, width: int = 20, invert: bool = False) -> str:
    """ASCII progress bar with value label."""
    pct = value if not invert else 1.0 - value
    filled = int(pct * width)
    bar_str = "█" * filled + "░" * (width - filled)
    return f"{bar_str} {value:.3f}"


class REFReportRenderer:
    """Renders a full terminal report for a RecognitiveScore."""

    def __init__(self, console: Console = None):
        self.console = console or Console(highlight=False)

    def render(self, score: RecognitiveScore, verbose: bool = False) -> None:
        c = self.console
        verdict_color, verdict_icon = COLORS.get(
            score.verdict, ("white", "?")
        )

        # ── Title banner ──────────────────────────────────────────────────────
        c.rule(style="dim cyan")
        title = Text()
        title.append("  REF ·  RECOGNITIVE EQUATION FRAMEWORK\n",
                     style="bold cyan")
        title.append(f"  {score.document_name}\n", style="bold white")
        title.append(
            f"  {score.word_count:,} words  ·  "
            f"{score.page_count} pages  ·  "
            f"{score.equation_count} equations  ·  "
            f"{score.heading_count} headings",
            style="dim"
        )
        c.print(Panel(title, border_style="dim cyan", padding=(0, 2)))

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict_text = Text()
        verdict_text.append(f"\n  {verdict_icon}  VERDICT:  ", style="bold dim")
        verdict_text.append(f"{score.verdict}",
                            style=f"bold {verdict_color}")
        verdict_text.append(f"  [{score.overall_score():.3f}]\n",
                            style=f"{verdict_color}")
        c.print(verdict_text)

        # ── Validity flags ────────────────────────────────────────────────────
        flags = score.validity_flags()
        flag_line = Text("  Validity criteria:  ", style="dim")
        criteria_map = {
            "recognition_ok":  f"Rec>{0.45:.2f}",
            "fidelity_ok":     f"ℰ<{0.55:.2f}",
            "hierarchy_ok":    f"ℋ<{0.40:.2f}",
            "concision_ok":    f"C>{0.30:.2f}",
        }
        for key, label in criteria_map.items():
            passed = flags[key]
            flag_line.append(
                f"  {'✓' if passed else '✗'} {label}",
                style="green" if passed else "red"
            )
        c.print(flag_line)
        c.print()

        # ── Five-axis radar table ─────────────────────────────────────────────
        c.rule("OPERATOR SCORES", style="dim")
        c.print()

        radar = score.radar_values()
        labels = {
            "Recognition":  ("Rec",   "Semantic + state alignment with S₀",       False),
            "Fidelity":     ("ℰ",     "1 − distortion (higher = less distortion)", False),
            "Egalitarian":  ("E_g",   "1 − hierarchy penalty",                     False),
            "Concision":    ("C",     "Information density I(T)/L(T)",             False),
            "Grounding":    ("ψ₁",   "Sensory / physical grounding signal",        False),
        }

        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("Axis",    style="bold cyan", width=14)
        table.add_column("Symbol",  style="dim",       width=6)
        table.add_column("Score",   style="white",     width=8)
        table.add_column("Bar",     style="white",     width=32)
        table.add_column("Description", style="dim",  width=40)

        for name, (symbol, desc, inv) in labels.items():
            val = radar[name]
            color = score_color(val, invert=inv)
            table.add_row(
                name, symbol,
                Text(f"{val:.3f}", style=f"bold {color}"),
                Text(bar(val), style=color),
                desc
            )

        c.print(table)

        # ── Quaternion state ──────────────────────────────────────────────────
        c.rule("QUATERNION STATE  Ψ ∈ ℍ⁴", style="dim")
        c.print()
        if score.initial_psi and score.final_psi:
            q_table = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
            q_table.add_column("Axis",    style="bold cyan", width=26)
            q_table.add_column("Ψ_init",  width=12)
            q_table.add_column("Ψ_final", width=12)
            q_table.add_column("Δ",       width=12)

            labels_psi = [
                ("ψ₀  Executive Control",   0),
                ("ψ₁  Sensory Grounding",   1),
                ("ψ₂  Memory Depth",        2),
                ("ψ₃  Affective Valence",   3),
            ]
            for label, i in labels_psi:
                init_v  = score.initial_psi.components[i]
                final_v = score.final_psi.components[i]
                delta   = final_v - init_v
                delta_str = f"{delta:+.4f}"
                dcolor = "green" if abs(delta) < 0.2 else "yellow" if abs(delta) < 0.5 else "red"
                q_table.add_row(
                    label,
                    f"{init_v:.4f}",
                    f"{final_v:.4f}",
                    Text(delta_str, style=dcolor)
                )

            c.print(q_table)
            if score.state_displacement > 0.4:
                disp_label = "HIGH — document significantly perturbs internal state"
            elif score.state_displacement > 0.2:
                disp_label = "MODERATE"
            else:
                disp_label = "LOW — state preserved"
            c.print(
                f"  State displacement ∥ΔΨ∥ = "
                f"{score.state_displacement:.4f}  ({disp_label})",
                style="dim")
            c.print()

        # ── Aggregate metrics ─────────────────────────────────────────────────
        c.rule("AGGREGATE METRICS", style="dim")
        c.print()

        m_table = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
        m_table.add_column(width=28)
        m_table.add_column(width=12)
        m_table.add_column(width=28)
        m_table.add_column(width=12)

        def metric_cell(label: str, value: float, invert: bool = False):
            color = score_color(value, invert)
            return Text(label, style="dim"), Text(f"{value:.4f}", style=f"bold {color}")

        rows = [
            ("Mean Recognition",  score.mean_recognition,  False,
             "Mean Concision",    score.mean_concision,     False),
            ("Mean Fidelity Δ",  score.mean_fidelity,      True,
             "Mean Grounding",    score.mean_grounding,     False),
            ("Mean Hierarchy",   score.mean_hierarchy,      True,
             "Mean Affect",       score.mean_affect,        False),
            ("Mean Egalitarian", score.mean_egalitarian,    False,
             "Temporal Mode τ",  (score.mean_temporal+1)/2, False),
        ]

        for l1, v1, i1, l2, v2, i2 in rows:
            c1, v_c1 = metric_cell(l1, v1, i1)
            c2, v_c2 = metric_cell(l2, v2, i2)
            m_table.add_row(c1, v_c1, c2, v_c2)

        c.print(m_table)
        c.print(f"  Temporal orientation:  "
                f"{_temporal_label(score.mean_temporal)}", style="dim")
        c.print()

        # ── Section breakdown ─────────────────────────────────────────────────
        if score.section_summary:
            c.rule("SECTION ANALYSIS", style="dim")
            c.print()
            sec_table = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
            sec_table.add_column("pg", style="dim", width=4)
            sec_table.add_column("Section",      style="cyan", width=52)
            sec_table.add_column("Rec",  width=8)
            sec_table.add_column("Hier", width=8)
            sec_table.add_column("Conc", width=8)

            for sec in score.section_summary:
                rc = score_color(sec["recognition"])
                hc = score_color(sec["hierarchy"], invert=True)
                cc = score_color(sec["concision"])
                sec_table.add_row(
                    str(sec["page"]),
                    sec["heading"],
                    Text(f"{sec['recognition']:.3f}", style=rc),
                    Text(f"{sec['hierarchy']:.3f}",   style=hc),
                    Text(f"{sec['concision']:.3f}",   style=cc),
                )
            c.print(sec_table)
            c.print()

        # ── Chunk detail (verbose mode) ───────────────────────────────────────
        if verbose and score.chunk_scores:
            c.rule("CHUNK DETAIL  (top 10 by distortion)", style="dim")
            c.print()
            top_chunks = sorted(score.chunk_scores,
                                key=lambda x: x.fidelity, reverse=True)[:10]
            for cs in top_chunks:
                chunk_color = score_color(cs.recognition)
                c.print(
                    f"  [{cs.chunk_index:03d}] "
                    f"Rec={cs.recognition:.3f}  ℰ={cs.fidelity:.3f}  "
                    f"ℋ={cs.hierarchy:.3f}  C={cs.concision:.3f}",
                    style=chunk_color
                )
                c.print(f"        {cs.text_preview}", style="dim")
            c.print()

        # ── Final Ψ state summary ─────────────────────────────────────────────
        if score.final_psi:
            c.rule("FINAL STATE INTERPRETATION", style="dim")
            c.print()
            axes = score.final_psi.axis_labels()
            interpretations = {
                "executive_control": ("  ψ₀ Executive Control ", "Focus/inhibition capacity after reading"),
                "sensory_grounding": ("  ψ₁ Sensory Grounding  ", "Physical-reality anchor strength"),
                "memory_depth":      ("  ψ₂ Memory Depth       ", "Contextual depth / prior knowledge activated"),
                "affective_valence": ("  ψ₃ Affective Valence  ", "Emotional charge residual"),
            }
            for key, (label, desc) in interpretations.items():
                val = axes[key]
                color = score_color(abs(val))
                bar_display = bar(max(0, (val + 1) / 2))
                c.print(f"{label}: {val:+.4f}  {bar_display}",
                        style=f"bold {color}")
                c.print(f"           └─ {desc}", style="dim")
            c.print()

        c.rule(style="dim cyan")
        c.print()


def _temporal_label(tau: float) -> str:
    if tau < -0.3:  return "PAST-ORIENTED   (historical anchoring, retrospective framing)"
    if tau >  0.3:  return "FUTURE-ORIENTED (anticipatory framing, speculative branching)"
    return "PRESENT-ANCHORED  (now-focused, grounding in immediate experience)"
