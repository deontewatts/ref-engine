"""
REF Differential Analyser
--------------------------
Computes pairwise contrasts between scored documents, revealing
which axis separates documents most cleanly and where operator
trajectories diverge.

This is the engine behind the compare command's discriminant insight.
Designed for integration with n8n / Make automation pipelines via
JSON output.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from ref_engine.scoring_engine import RecognitiveScore


@dataclass
class AxisProfile:
    """Per-axis descriptor for a document."""
    name: str
    recognition:  float
    fidelity:     float   # distortion  (lower = better)
    hierarchy:    float
    concision:    float
    grounding:    float
    affect:       float
    temporal:     float
    egalitarian:  float
    overall:      float
    verdict:      str

    def as_vector(self) -> np.ndarray:
        """8-dim feature vector for distance computation."""
        return np.array([
            self.recognition,
            1 - self.fidelity,
            1 - self.hierarchy,
            self.concision,
            self.grounding,
            self.affect,
            (self.temporal + 1) / 2,
            self.egalitarian,
        ])


@dataclass
class PairwiseContrast:
    """Contrast between two documents."""
    doc_a: str
    doc_b: str
    euclidean_distance: float
    cosine_similarity:  float
    divergent_axes:     List[Tuple[str, float]]   # (axis_name, |delta|) sorted desc
    summary:            str


@dataclass
class CorpusAnalysis:
    """Full corpus-level differential analysis."""
    profiles:           List[AxisProfile]
    pairwise:           List[PairwiseContrast]
    axis_variances:     Dict[str, float]
    top_discriminant:   str
    centroid:           Dict[str, float]
    outliers:           List[str]   # docs >1 std from centroid


AXIS_NAMES = [
    "recognition", "fidelity_inv", "hierarchy_inv",
    "concision", "grounding", "affect", "temporal_norm", "egalitarian"
]

AXIS_LABELS = {
    "recognition":    "Recognition",
    "fidelity_inv":   "1−ℰ (Fidelity)",
    "hierarchy_inv":  "1−ℋ (Egalitarian)",
    "concision":      "Concision C",
    "grounding":      "Grounding ψ₁",
    "affect":         "Affect ψ₃",
    "temporal_norm":  "Temporal τ",
    "egalitarian":    "Egalitarian E_g",
}


class DifferentialAnalyser:

    def analyse(self, scores: List[RecognitiveScore]) -> CorpusAnalysis:
        profiles = [self._to_profile(s) for s in scores]
        vectors  = np.stack([p.as_vector() for p in profiles])

        # Axis variances
        variances = {AXIS_NAMES[i]: float(np.var(vectors[:, i]))
                     for i in range(len(AXIS_NAMES))}
        top_discriminant = max(variances, key=variances.get)

        # Centroid
        centroid_vec = vectors.mean(axis=0)
        centroid = {AXIS_NAMES[i]: float(centroid_vec[i])
                    for i in range(len(AXIS_NAMES))}

        # Outliers: docs > 1 std from centroid
        dists_from_centroid = np.linalg.norm(vectors - centroid_vec, axis=1)
        std = np.std(dists_from_centroid)
        mean = np.mean(dists_from_centroid)
        outliers = [profiles[i].name for i in range(len(profiles))
                    if dists_from_centroid[i] > mean + std]

        # Pairwise contrasts
        pairwise = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                contrast = self._pairwise_contrast(
                    profiles[i], profiles[j], vectors[i], vectors[j]
                )
                pairwise.append(contrast)

        # Sort by distance (most different pairs first)
        pairwise.sort(key=lambda c: c.euclidean_distance, reverse=True)

        return CorpusAnalysis(
            profiles=profiles,
            pairwise=pairwise,
            axis_variances=variances,
            top_discriminant=top_discriminant,
            centroid=centroid,
            outliers=outliers,
        )

    def _to_profile(self, s: RecognitiveScore) -> AxisProfile:
        return AxisProfile(
            name=s.document_name,
            recognition=s.mean_recognition,
            fidelity=s.mean_fidelity,
            hierarchy=s.mean_hierarchy,
            concision=s.mean_concision,
            grounding=s.mean_grounding,
            affect=s.mean_affect,
            temporal=s.mean_temporal,
            egalitarian=s.mean_egalitarian,
            overall=s.overall_score(),
            verdict=s.verdict,
        )

    def _pairwise_contrast(self,
                           a: AxisProfile, b: AxisProfile,
                           va: np.ndarray, vb: np.ndarray) -> PairwiseContrast:
        delta = vb - va
        euclidean = float(np.linalg.norm(delta))

        # Cosine similarity
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a > 0 and norm_b > 0:
            cos_sim = float(np.dot(va, vb) / (norm_a * norm_b))
        else:
            cos_sim = 0.0

        # Divergent axes sorted by absolute delta
        divergent = sorted(
            [(AXIS_NAMES[i], abs(float(delta[i])))
             for i in range(len(AXIS_NAMES))],
            key=lambda x: x[1], reverse=True
        )[:4]

        # Human-readable summary
        top_axis = AXIS_LABELS.get(divergent[0][0], divergent[0][0])
        summary = (
            f"{a.name[:28]} vs {b.name[:28]}: "
            f"distance={euclidean:.4f}, cosine={cos_sim:.4f}. "
            f"Primary divergence on {top_axis} (Δ={divergent[0][1]:.4f})."
        )

        return PairwiseContrast(
            doc_a=a.name, doc_b=b.name,
            euclidean_distance=euclidean,
            cosine_similarity=cos_sim,
            divergent_axes=divergent,
            summary=summary,
        )

    def render_analysis(self, analysis: CorpusAnalysis,
                        console) -> None:
        from rich.table import Table
        from rich.text import Text
        from rich import box

        console.rule("DIFFERENTIAL CORPUS ANALYSIS", style="bold cyan")
        console.print()

        # Axis variance table
        console.print("  [bold]Axis Discriminability[/]  "
                      "(variance across corpus — higher = more separating)\n")
        var_table = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
        var_table.add_column("Axis",          style="cyan", width=26)
        var_table.add_column("Variance",      width=10)
        var_table.add_column("Bar",           width=30)
        var_table.add_column("Top Separator?",width=14)

        max_var = max(analysis.axis_variances.values()) + 1e-9
        for ax, var in sorted(analysis.axis_variances.items(),
                               key=lambda x: x[1], reverse=True):
            bar_w = int((var / max_var) * 24)
            bar_str = "█" * bar_w + "░" * (24 - bar_w)
            is_top = "★ YES" if ax == analysis.top_discriminant else ""
            color = "bright_green" if ax == analysis.top_discriminant else \
                    "yellow" if var > max_var * 0.3 else "dim"
            var_table.add_row(
                AXIS_LABELS.get(ax, ax),
                Text(f"{var:.5f}", style=color),
                Text(bar_str, style=color),
                Text(is_top, style="bright_green bold"),
            )
        console.print(var_table)
        console.print()

        # Corpus centroid
        console.print("  [bold]Corpus Centroid[/]  "
                      "(average position across all documents)\n")
        for ax, val in analysis.centroid.items():
            label = AXIS_LABELS.get(ax, ax)
            console.print(f"    {label:<28}  {val:.4f}", style="dim")
        console.print()

        # Outliers
        if analysis.outliers:
            console.print(
                "  [bold yellow]Outlier documents[/] "
                "(>1σ from centroid):  "
                + ",  ".join("[yellow]{}[/]".format(o) for o in analysis.outliers)
            )
            console.print()

        # Top 5 most-divergent pairs
        console.print("  [bold]Most Divergent Document Pairs[/]\n")
        for contrast in analysis.pairwise[:5]:
            cos_color = "green" if contrast.cosine_similarity > 0.9 else \
                        "yellow" if contrast.cosine_similarity > 0.7 else "red"
            console.print(
                f"  [cyan]{contrast.doc_a[:24]:24}[/]  ↔  "
                f"[cyan]{contrast.doc_b[:24]:24}[/]"
            )
            console.print(
                f"    dist={contrast.euclidean_distance:.4f}  "
                f"cos=[{cos_color}]{contrast.cosine_similarity:.4f}[/]  "
                f"top-div: {AXIS_LABELS.get(contrast.divergent_axes[0][0], '')}"
                f"  Δ={contrast.divergent_axes[0][1]:.4f}",
                style="dim"
            )
        console.print()
