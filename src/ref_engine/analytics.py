"""
REF Analytics Engine
---------------------
Advanced agentic analysis beyond single-document scoring:

  1. DiffEngine        — compare two documents across all axes, identify
                         divergence points and structural deltas
  2. TemporalEvolution — track how Ψ evolves chunk-by-chunk across a document;
                         produces a state trajectory suitable for plotting
  3. SignalExtractor   — pull high-recognition / high-manipulation passages
                         from a document for targeted review
  4. CrossCorpusIndex  — build a searchable index of recognitive scores across
                         an entire corpus; supports nearest-neighbor lookup by
                         operator profile
  5. OperatorDecomposer— break a single score into its five operator
                         contributions with attribution to specific passages
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ref_engine.scoring_engine import RecognitiveScore


# ─── 1. Diff Engine ───────────────────────────────────────────────────────────

@dataclass
class DocumentDiff:
    """
    Axis-by-axis delta between two documents.
    Positive Δ means doc_b scores higher on that axis.
    """
    doc_a: str
    doc_b: str
    delta_recognition:   float
    delta_fidelity:      float   # positive = doc_b has LESS distortion
    delta_hierarchy:     float   # positive = doc_b has MORE hierarchy
    delta_concision:     float
    delta_grounding:     float
    delta_affect:        float
    delta_temporal:      float
    delta_overall:       float
    psi_divergence:      float   # ∥Ψ_a_final − Ψ_b_final∥
    dominant_difference: str     # which axis shows largest absolute delta
    interpretation:      str


class DiffEngine:
    """Compare two RecognitiveScores and produce a structured diff."""

    def diff(self, score_a: RecognitiveScore,
             score_b: RecognitiveScore) -> DocumentDiff:
        axes = {
            "recognition": (score_b.mean_recognition - score_a.mean_recognition),
            "fidelity":    (score_a.mean_fidelity    - score_b.mean_fidelity),   # inverted: less distortion = better
            "hierarchy":   (score_b.mean_hierarchy    - score_a.mean_hierarchy),
            "concision":   (score_b.mean_concision    - score_a.mean_concision),
            "grounding":   (score_b.mean_grounding    - score_a.mean_grounding),
            "affect":      (score_b.mean_affect       - score_a.mean_affect),
            "temporal":    (score_b.mean_temporal     - score_a.mean_temporal),
        }
        dominant = max(axes, key=lambda k: abs(axes[k]))

        # Quaternion state divergence
        psi_div = 0.0
        if score_a.final_psi and score_b.final_psi:
            psi_div = float(np.linalg.norm(
                score_a.final_psi.components - score_b.final_psi.components
            ))

        delta_overall = score_b.overall_score() - score_a.overall_score()
        interpretation = self._interpret(axes, dominant, delta_overall)

        return DocumentDiff(
            doc_a            = score_a.document_name,
            doc_b            = score_b.document_name,
            delta_recognition  = round(axes["recognition"], 4),
            delta_fidelity     = round(axes["fidelity"],    4),
            delta_hierarchy    = round(axes["hierarchy"],   4),
            delta_concision    = round(axes["concision"],   4),
            delta_grounding    = round(axes["grounding"],   4),
            delta_affect       = round(axes["affect"],      4),
            delta_temporal     = round(axes["temporal"],    4),
            delta_overall      = round(delta_overall,       4),
            psi_divergence     = round(psi_div,             4),
            dominant_difference= dominant,
            interpretation     = interpretation,
        )

    def _interpret(self, axes: Dict[str, float],
                   dominant: str, delta_overall: float) -> str:
        direction = "higher" if delta_overall > 0 else "lower"
        abs_dom = abs(axes[dominant])

        interp_map = {
            "recognition": (
                f"Doc B has {'stronger' if axes['recognition'] > 0 else 'weaker'} "
                f"semantic-state alignment (Δ={axes['recognition']:+.3f}). "
                f"This is the primary differentiator (Δ={abs_dom:.3f})."
            ),
            "fidelity": (
                f"Doc B introduces {'less' if axes['fidelity'] > 0 else 'more'} "
                f"state distortion (Δ={axes['fidelity']:+.3f}). "
                f"Fidelity is the primary axis of divergence."
            ),
            "hierarchy": (
                f"Doc B has {'more' if axes['hierarchy'] > 0 else 'less'} "
                f"authority-asymmetric framing (Δ={axes['hierarchy']:+.3f}). "
                f"Hierarchy framing is the largest differentiator."
            ),
            "concision": (
                f"Doc B is {'denser' if axes['concision'] > 0 else 'more dilute'} "
                f"in information per token (Δ={axes['concision']:+.3f}). "
                f"Concision is the primary axis of divergence."
            ),
            "grounding": (
                f"Doc B is {'more' if axes['grounding'] > 0 else 'less'} "
                f"physically grounded (Δ={axes['grounding']:+.3f}). "
                f"Sensory anchoring is the primary differentiator."
            ),
            "affect": (
                f"Doc B carries {'higher' if axes['affect'] > 0 else 'lower'} "
                f"affective load (Δ={axes['affect']:+.3f}). "
                f"Emotional resonance is the dominant axis."
            ),
            "temporal": (
                f"Doc B is more {'future-' if axes['temporal'] > 0 else 'past-'}"
                f"oriented (Δ={axes['temporal']:+.3f}). "
                f"Temporal framing is the primary differentiator."
            ),
        }

        base = interp_map.get(dominant, "Complex multi-axis difference.")
        return f"Doc B scores {direction} overall (Δ={delta_overall:+.3f}). {base}"


# ─── 2. Temporal Evolution ────────────────────────────────────────────────────

@dataclass
class StateTrajectory:
    """
    The full Ψ trajectory across a document's chunks.
    Captures how the internal state evolves as reading progresses.
    """
    document_name: str
    chunk_indices: List[int]
    psi0_trace: List[float]    # Executive Control over time
    psi1_trace: List[float]    # Sensory Grounding over time
    psi2_trace: List[float]    # Memory Depth over time
    psi3_trace: List[float]    # Affective Valence over time
    recognition_trace: List[float]
    hierarchy_trace:   List[float]
    fidelity_trace:    List[float]

    def turning_points(self, axis: str = "recognition") -> List[int]:
        """
        Find chunk indices where the trajectory reverses direction
        (local minima/maxima) — critical transition points in the document.
        """
        trace_map = {
            "recognition": self.recognition_trace,
            "hierarchy":   self.hierarchy_trace,
            "fidelity":    self.fidelity_trace,
            "psi0":        self.psi0_trace,
            "psi3":        self.psi3_trace,
        }
        trace = trace_map.get(axis, self.recognition_trace)
        if len(trace) < 3:
            return []

        turns = []
        for i in range(1, len(trace) - 1):
            if (trace[i] > trace[i-1] and trace[i] > trace[i+1]) or \
               (trace[i] < trace[i-1] and trace[i] < trace[i+1]):
                turns.append(self.chunk_indices[i])
        return turns

    def volatility(self, axis: str = "recognition") -> float:
        """
        Mean absolute difference between successive values on an axis.
        High volatility = document has unstable recognitive profile.
        """
        trace_map = {
            "recognition": self.recognition_trace,
            "hierarchy":   self.hierarchy_trace,
            "fidelity":    self.fidelity_trace,
        }
        trace = trace_map.get(axis, self.recognition_trace)
        if len(trace) < 2:
            return 0.0
        diffs = [abs(trace[i] - trace[i-1]) for i in range(1, len(trace))]
        return float(np.mean(diffs))

    def to_dict(self) -> dict:
        return {
            "document": self.document_name,
            "chunks":   self.chunk_indices,
            "traces": {
                "psi0_executive":  self.psi0_trace,
                "psi1_grounding":  self.psi1_trace,
                "psi2_memory":     self.psi2_trace,
                "psi3_affect":     self.psi3_trace,
                "recognition":     self.recognition_trace,
                "hierarchy":       self.hierarchy_trace,
                "fidelity_dist":   self.fidelity_trace,
            },
            "turning_points": {
                "recognition": self.turning_points("recognition"),
                "hierarchy":   self.turning_points("hierarchy"),
            },
            "volatility": {
                "recognition": round(self.volatility("recognition"), 4),
                "hierarchy":   round(self.volatility("hierarchy"),   4),
                "fidelity":    round(self.volatility("fidelity"),    4),
            }
        }


class TemporalEvolution:
    """Extract the full state trajectory from a scored document."""

    def extract(self, score: RecognitiveScore) -> StateTrajectory:
        chunks = score.chunk_scores
        if not chunks:
            return StateTrajectory(
                document_name=score.document_name,
                chunk_indices=[], psi0_trace=[], psi1_trace=[],
                psi2_trace=[], psi3_trace=[], recognition_trace=[],
                hierarchy_trace=[], fidelity_trace=[]
            )

        return StateTrajectory(
            document_name    = score.document_name,
            chunk_indices    = [c.chunk_index for c in chunks],
            psi0_trace       = [float(c.psi_after.components[0]) for c in chunks],
            psi1_trace       = [float(c.psi_after.components[1]) for c in chunks],
            psi2_trace       = [float(c.psi_after.components[2]) for c in chunks],
            psi3_trace       = [float(c.psi_after.components[3]) for c in chunks],
            recognition_trace= [c.recognition for c in chunks],
            hierarchy_trace  = [c.hierarchy   for c in chunks],
            fidelity_trace   = [c.fidelity    for c in chunks],
        )


# ─── 3. Signal Extractor ─────────────────────────────────────────────────────

@dataclass
class ExtractedSignal:
    """A high-signal passage flagged by the REF engine."""
    signal_type: str     # "HIGH_RECOGNITION" | "HIGH_MANIPULATION" | "HIERARCHY_PEAK"
    chunk_index: int
    score:       float
    text_preview: str
    axis_values: Dict[str, float]


class SignalExtractor:
    """
    Identifies and extracts the highest-signal passages from a scored document.
    Useful for targeted review: instead of reading everything, an analyst can
    see exactly which passages drive the recognitive or manipulative profile.
    """

    def extract(self, score: RecognitiveScore,
                top_n: int = 5) -> Dict[str, List[ExtractedSignal]]:
        chunks = score.chunk_scores
        signals = defaultdict(list)

        # High recognition passages
        for cs in sorted(chunks, key=lambda c: c.recognition, reverse=True)[:top_n]:
            signals["HIGH_RECOGNITION"].append(ExtractedSignal(
                signal_type  = "HIGH_RECOGNITION",
                chunk_index  = cs.chunk_index,
                score        = round(cs.recognition, 4),
                text_preview = cs.text_preview,
                axis_values  = {"recognition": cs.recognition,
                                "hierarchy": cs.hierarchy, "concision": cs.concision}
            ))

        # High manipulation / distortion passages
        for cs in sorted(chunks, key=lambda c: c.fidelity, reverse=True)[:top_n]:
            if cs.fidelity > 0.05:  # only flag meaningful distortion
                signals["HIGH_DISTORTION"].append(ExtractedSignal(
                    signal_type  = "HIGH_DISTORTION",
                    chunk_index  = cs.chunk_index,
                    score        = round(cs.fidelity, 4),
                    text_preview = cs.text_preview,
                    axis_values  = {"fidelity_dist": cs.fidelity,
                                    "hierarchy": cs.hierarchy,
                                    "synthetic_p": cs.features.synthetic_pressure}
                ))

        # Hierarchy peaks
        for cs in sorted(chunks, key=lambda c: c.hierarchy, reverse=True)[:top_n]:
            if cs.hierarchy > 0.20:
                signals["HIERARCHY_PEAK"].append(ExtractedSignal(
                    signal_type  = "HIERARCHY_PEAK",
                    chunk_index  = cs.chunk_index,
                    score        = round(cs.hierarchy, 4),
                    text_preview = cs.text_preview,
                    axis_values  = {"hierarchy": cs.hierarchy,
                                    "egalitarian": cs.egalitarian,
                                    "affect": cs.features.affective_valence}
                ))

        return dict(signals)


# ─── 4. Cross-Corpus Index ────────────────────────────────────────────────────

@dataclass
class CorpusEntry:
    document_name: str
    overall_score: float
    operator_vector: np.ndarray   # [rec, 1-fid, 1-hier, conc, grnd, aff]
    verdict: str


class CrossCorpusIndex:
    """
    Builds a searchable vector index of all scored documents.
    Supports nearest-neighbor lookup: given a query document,
    find the corpus members with the most similar operator profile.
    """

    def __init__(self):
        self.entries: List[CorpusEntry] = []

    def add(self, score: RecognitiveScore) -> None:
        vec = np.array([
            score.mean_recognition,
            1.0 - score.mean_fidelity,
            1.0 - score.mean_hierarchy,
            score.mean_concision,
            score.mean_grounding,
            score.mean_affect,
        ])
        self.entries.append(CorpusEntry(
            document_name    = score.document_name,
            overall_score    = score.overall_score(),
            operator_vector  = vec,
            verdict          = score.verdict,
        ))

    def nearest(self, score: RecognitiveScore,
                k: int = 3) -> List[Tuple[str, float]]:
        """
        Return the k most similar documents by cosine similarity
        of operator vectors.
        """
        if not self.entries:
            return []
        query_vec = np.array([
            score.mean_recognition,
            1.0 - score.mean_fidelity,
            1.0 - score.mean_hierarchy,
            score.mean_concision,
            score.mean_grounding,
            score.mean_affect,
        ])
        sims = []
        for entry in self.entries:
            if entry.document_name == score.document_name:
                continue
            cos_sim = float(
                np.dot(query_vec, entry.operator_vector) /
                (np.linalg.norm(query_vec) * np.linalg.norm(entry.operator_vector) + 1e-9)
            )
            sims.append((entry.document_name, round(cos_sim, 4)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def cluster_by_verdict(self) -> Dict[str, List[str]]:
        """Group all indexed documents by their verdict."""
        clusters = defaultdict(list)
        for entry in self.entries:
            clusters[entry.verdict].append(entry.document_name)
        return dict(clusters)

    def operator_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute mean and std of each operator across the corpus."""
        if not self.entries:
            return {}
        axis_names = ["recognition", "1-fidelity", "1-hierarchy",
                      "concision", "grounding", "affect"]
        matrix = np.stack([e.operator_vector for e in self.entries])
        stats = {}
        for i, name in enumerate(axis_names):
            col = matrix[:, i]
            stats[name] = {
                "mean": round(float(np.mean(col)), 4),
                "std":  round(float(np.std(col)),  4),
                "min":  round(float(np.min(col)),  4),
                "max":  round(float(np.max(col)),  4),
            }
        return stats

    def to_dict(self) -> dict:
        return {
            "corpus_size": len(self.entries),
            "documents": [
                {
                    "name": e.document_name,
                    "overall": round(e.overall_score, 4),
                    "verdict": e.verdict,
                    "vector": [round(float(v), 4) for v in e.operator_vector],
                }
                for e in self.entries
            ],
            "statistics": self.operator_statistics(),
            "clusters": self.cluster_by_verdict(),
        }


# ─── 5. Operator Decomposer ───────────────────────────────────────────────────

@dataclass
class OperatorDecomposition:
    """Attribution of each operator's contribution to the overall score."""
    document_name: str
    operator_contributions: Dict[str, float]   # name → weighted contribution
    top_driver: str                             # operator with highest contribution
    bottom_driver: str                          # operator that most limits score
    bottleneck_passages: List[str]              # text previews of lowest-scoring chunks
    enhancement_passages: List[str]             # text previews of highest-scoring chunks
    prescription: str                           # what to change to improve score


class OperatorDecomposer:
    """
    Decompose an overall score into its five operator contributions
    and identify actionable improvement levers.
    """

    WEIGHTS = {
        "Recognition": 0.25,
        "Fidelity":    0.30,
        "Egalitarian": 0.25,
        "Concision":   0.10,
        "Grounding":   0.10,
    }

    def decompose(self, score: RecognitiveScore) -> OperatorDecomposition:
        radar = score.radar_values()
        contributions = {k: round(v * self.WEIGHTS[k], 4) for k, v in radar.items()}
        top    = max(contributions, key=contributions.get)
        bottom = min(contributions, key=contributions.get)

        # Bottleneck passages: lowest score on bottom_driver axis
        axis_map = {
            "Recognition": "recognition",
            "Fidelity":    "fidelity",
            "Egalitarian": "hierarchy",
            "Concision":   "concision",
            "Grounding":   "recognition",  # proxy
        }
        bottom_attr = axis_map[bottom]

        if bottom_attr == "fidelity":
            bottleneck_chunks = sorted(score.chunk_scores,
                                       key=lambda c: c.fidelity, reverse=True)[:3]
        elif bottom_attr == "hierarchy":
            bottleneck_chunks = sorted(score.chunk_scores,
                                       key=lambda c: c.hierarchy, reverse=True)[:3]
        elif bottom_attr == "concision":
            bottleneck_chunks = sorted(score.chunk_scores,
                                       key=lambda c: c.concision)[:3]
        else:
            bottleneck_chunks = sorted(score.chunk_scores,
                                       key=lambda c: c.recognition)[:3]

        top_attr = axis_map[top]
        if top_attr == "recognition":
            enhance_chunks = sorted(score.chunk_scores,
                                    key=lambda c: c.recognition, reverse=True)[:3]
        else:
            enhance_chunks = sorted(score.chunk_scores,
                                    key=lambda c: c.concision, reverse=True)[:3]

        prescription = self._prescribe(bottom, radar[bottom])

        return OperatorDecomposition(
            document_name          = score.document_name,
            operator_contributions = contributions,
            top_driver             = top,
            bottom_driver          = bottom,
            bottleneck_passages    = [c.text_preview[:120] for c in bottleneck_chunks],
            enhancement_passages   = [c.text_preview[:120] for c in enhance_chunks],
            prescription           = prescription,
        )

    def _prescribe(self, bottom_axis: str, current_value: float) -> str:
        gap = 1.0 - current_value
        prescriptions = {
            "Recognition": (
                f"Recognition scores {current_value:.3f} (gap={gap:.3f}). "
                "Improve semantic coherence: increase bigram repetition, "
                "add connective tissue (therefore/because/thus), "
                "tighten topical focus across chunks."
            ),
            "Fidelity": (
                f"Fidelity scores {current_value:.3f} (gap={gap:.3f}). "
                "Reduce state distortion: lower affective arousal language, "
                "reduce urgency/threat framing, add empirical hedging."
            ),
            "Egalitarian": (
                f"Egalitarian scores {current_value:.3f} (gap={gap:.3f}). "
                "Reduce hierarchy: replace 'you must/should' with 'we can/notice', "
                "shift from epistemic authority ('I will teach you') to shared inquiry, "
                "add more egalitarian markers (together, as equals, open source)."
            ),
            "Concision": (
                f"Concision scores {current_value:.3f} (gap={gap:.3f}). "
                "Increase information density: shorten average sentence length (<15 words), "
                "remove redundant phrases, increase lexical variety."
            ),
            "Grounding": (
                f"Grounding scores {current_value:.3f} (gap={gap:.3f}). "
                "Increase physical anchoring: add body/sensory/spatial language "
                "(breath, hands, here, now, ground), concrete examples, "
                "direct experiential references."
            ),
        }
        return prescriptions.get(bottom_axis, "No specific prescription available.")
