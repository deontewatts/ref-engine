"""
REF Scoring Engine
------------------
The central computation layer. Given a ParsedDocument, produces a
RecognitiveScore with all five operator values and a final validity verdict.

Recognitive Validity Criterion:
    F is RECOGNITIVE iff:
        Rec(F,Ψ₀)  > θ_r  = 0.45   (recognition threshold)
        ℰ_F        < θ_f  = 0.55   (fidelity distortion threshold)
        ℋ(T)       < θ_h  = 0.40   (hierarchy penalty threshold)
        C(T)       > θ_c  = 0.30   (concision threshold)

    F is MANIPULATIVE iff:
        ℰ_F > 0.65 OR ℋ(T) > 0.60

    Otherwise: NEUTRAL / INDETERMINATE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from ref_engine.quaternion import (QuaternionState, recognitive_reset)
from ref_engine.feature_extractor import TextFeatureExtractor, TextFeatures
from ref_engine.file_parser import ParsedDocument


# ─── Thresholds ───────────────────────────────────────────────────────────────

THETA_R = 0.45   # minimum recognition score
THETA_F = 0.55   # maximum fidelity distortion (lower = better)
THETA_H = 0.40   # maximum hierarchy score
THETA_C = 0.30   # minimum concision density


# ─── Score Dataclass ──────────────────────────────────────────────────────────

@dataclass
class ChunkScore:
    """Recognitive scores for a single text chunk."""
    chunk_index: int
    text_preview: str
    features: TextFeatures
    psi_before: QuaternionState
    psi_after: QuaternionState
    recognition: float
    fidelity: float        # ℰ_F  (distortion, lower = better)
    hierarchy: float
    concision: float
    egalitarian: float


@dataclass
class RecognitiveScore:
    """
    Full recognitive analysis of a ParsedDocument.
    Aggregates across all chunks and produces per-document metrics.
    """
    document_name: str
    word_count: int
    page_count: int
    chunk_scores: List[ChunkScore] = field(default_factory=list)

    # Aggregate metrics
    mean_recognition:   float = 0.0
    mean_fidelity:      float = 0.0   # mean distortion (lower = better)
    mean_hierarchy:     float = 0.0
    mean_concision:     float = 0.0
    mean_egalitarian:   float = 0.0
    mean_affect:        float = 0.0
    mean_grounding:     float = 0.0
    mean_temporal:      float = 0.0

    # Quaternion state trajectory (source → final)
    initial_psi: Optional[QuaternionState] = None
    final_psi:   Optional[QuaternionState] = None
    state_displacement: float = 0.0  # ∥Ψ_final − Ψ_initial∥

    # Verdict
    verdict:     str = "UNSCORED"
    verdict_code: int = 0   # 1=recognitive, 0=neutral, -1=manipulative

    # Structural metrics
    equation_count:  int = 0
    heading_count:   int = 0
    section_summary: List[Dict] = field(default_factory=list)

    def is_recognitive(self) -> bool:
        return self.verdict_code == 1

    def is_manipulative(self) -> bool:
        return self.verdict_code == -1

    def validity_flags(self) -> Dict[str, bool]:
        """Which of the four validity criteria are satisfied?"""
        return {
            "recognition_ok":  self.mean_recognition > THETA_R,
            "fidelity_ok":     self.mean_fidelity     < THETA_F,
            "hierarchy_ok":    self.mean_hierarchy     < THETA_H,
            "concision_ok":    self.mean_concision     > THETA_C,
        }

    def radar_values(self) -> Dict[str, float]:
        """Five-axis radar chart values, all in [0,1]."""
        return {
            "Recognition":  self.mean_recognition,
            "Fidelity":     1.0 - self.mean_fidelity,   # invert: high=good
            "Egalitarian":  1.0 - self.mean_hierarchy,
            "Concision":    self.mean_concision,
            "Grounding":    self.mean_grounding,
        }

    def overall_score(self) -> float:
        """
        Single composite score ∈ [0,1].
        Weighted harmonic mean emphasizing fidelity and hierarchy.
        """
        r  = self.radar_values()
        weights = {"Recognition": 0.25, "Fidelity": 0.30,
                   "Egalitarian": 0.25, "Concision": 0.10, "Grounding": 0.10}
        num = sum(weights[k] for k in weights)
        weighted = sum(r[k] * weights[k] for k in weights)
        return weighted / num


# ─── Scoring Engine ───────────────────────────────────────────────────────────

class REFScoringEngine:
    """
    Agentically scores a ParsedDocument against the REF framework.
    
    Pipeline:
        1. Chunk document into analysis windows
        2. Extract TextFeatures per chunk
        3. Map features → QuaternionState Ψ
        4. Compute recognitive operators per chunk
        5. Evolve state across the document (Ψ_{k+1} = ℛ_F(Ψ_k))
        6. Aggregate scores, compute verdict
    """

    def __init__(self, chunk_size: int = 400, verbose: bool = False):
        self.extractor = TextFeatureExtractor()
        self.chunk_size = chunk_size
        self.verbose = verbose

    def score(self, doc: ParsedDocument) -> RecognitiveScore:
        result = RecognitiveScore(
            document_name=doc.filename,
            word_count=doc.word_count,
            page_count=doc.page_count,
            equation_count=doc.equation_count,
            heading_count=doc.heading_count,
        )

        chunks = doc.get_chunk(self.chunk_size)
        if not chunks:
            result.verdict = "EMPTY_DOCUMENT"
            return result

        # ── Initial state: S₀ baseline ──────────────────────────────────────
        psi = QuaternionState.baseline()
        result.initial_psi = psi

        chunk_scores = []

        for idx, chunk in enumerate(chunks):
            feat = self.extractor.extract(chunk)
            psi_before = psi

            # Map TextFeatures → QuaternionState
            psi_text = QuaternionState.from_features(
                attention = feat.attention_stability,
                grounding = feat.sensory_grounding,
                memory    = feat.memory_depth,
                affect    = feat.affective_valence,
            )

            # Apply file operator: this chunk's recognitive action
            # ℛ_F(Ψ) = recognitive_reset modulated by chunk features
            psi_after = self._apply_chunk_operator(psi, psi_text, feat)

            # Recognition: alignment between evolved state and source
            source = QuaternionState.baseline()
            recognition = source.fidelity(psi_after)

            # Fidelity distortion: how much did this chunk warp Ψ away from Ψ₀?
            fidelity_distortion = 1.0 - psi_before.fidelity(psi_after)

            # Hierarchy and concision from features
            hierarchy  = feat.hierarchy_score
            concision  = feat.concision_density
            egalitarian = feat.egalitarian_score()

            chunk_scores.append(ChunkScore(
                chunk_index  = idx,
                text_preview = chunk[:120].replace("\n", " ") + "…",
                features     = feat,
                psi_before   = psi_before,
                psi_after    = psi_after,
                recognition  = recognition,
                fidelity     = fidelity_distortion,
                hierarchy    = hierarchy,
                concision    = concision,
                egalitarian  = egalitarian,
            ))

            psi = psi_after  # carry state forward

        result.chunk_scores    = chunk_scores
        result.final_psi       = psi

        # State displacement: how far has the document moved Ψ from baseline?
        result.state_displacement = float(np.linalg.norm(
            psi.components - QuaternionState.baseline().components
        ))

        # Aggregate
        result.mean_recognition = float(np.mean([c.recognition  for c in chunk_scores]))
        result.mean_fidelity    = float(np.mean([c.fidelity      for c in chunk_scores]))
        result.mean_hierarchy   = float(np.mean([c.hierarchy     for c in chunk_scores]))
        result.mean_concision   = float(np.mean([c.concision     for c in chunk_scores]))
        result.mean_egalitarian = float(np.mean([c.egalitarian   for c in chunk_scores]))
        result.mean_affect      = float(np.mean([c.features.affective_valence
                                                  for c in chunk_scores]))
        result.mean_grounding   = float(np.mean([c.features.sensory_grounding
                                                  for c in chunk_scores]))
        result.mean_temporal    = float(np.mean([c.features.temporal_mode
                                                  for c in chunk_scores]))

        # Section summaries
        result.section_summary = self._summarize_sections(doc, chunk_scores)

        # Verdict
        result.verdict, result.verdict_code = self._classify(result)

        return result

    def _apply_chunk_operator(self,
                               psi: QuaternionState,
                               psi_text: QuaternionState,
                               feat: TextFeatures) -> QuaternionState:
        """
        ∂Ψ/∂t = α∇Ψ + β(Ψ×Ψ*) + γΦ_ext + δ𝒮_F(Ψ)
        
        Discretized as:
            Ψ_{t+1} = 0.6·Ψ_t + 0.3·Ψ_text + 0.1·S₀
            then apply Pause→Notice→Return if synthetic_pressure is high
        """
        # Blend current state with text-induced state (non-linear mixing)
        alpha, beta, gamma = 0.60, 0.30, 0.10
        blended = QuaternionState(
            alpha * psi.components +
            beta  * psi_text.components +
            gamma * QuaternionState.baseline().components
        )

        # Non-linear self-interaction term: β(Ψ×Ψ*)
        # psi.multiply(psi.conjugate()) is always real (|Ψ|²·1)
        # We use the norm as a stabilizing correction
        norm_correction = min(1.0, 1.0 / (blended.norm() + 1e-6))
        blended = QuaternionState(blended.components * norm_correction)

        # Apply recognitive reset if synthetic pressure is elevated
        if feat.synthetic_pressure > 0.5:
            blended = recognitive_reset(blended, feat.synthetic_pressure)

        return blended

    def _classify(self, result: RecognitiveScore) -> Tuple[str, int]:
        """
        Apply the four-criterion validity test.
        Returns (verdict_string, verdict_code).
        """
        flags = result.validity_flags()
        all_pass = all(flags.values())
        
        # Hard manipulation tests
        is_manipulative = (
            result.mean_fidelity  > 0.65 or
            result.mean_hierarchy > 0.60
        )

        if is_manipulative:
            return "MANIPULATIVE", -1
        elif all_pass:
            return "RECOGNITIVE", 1
        elif flags["recognition_ok"] and flags["fidelity_ok"]:
            return "PARTIALLY_RECOGNITIVE", 0
        elif not flags["fidelity_ok"] or not flags["hierarchy_ok"]:
            return "HIGH_DISTORTION", -1
        else:
            return "NEUTRAL", 0

    def _summarize_sections(self, doc: ParsedDocument,
                             chunk_scores: List[ChunkScore]) -> List[Dict]:
        """Build per-section score summaries from section headings."""
        summary = []
        for sec in doc.sections[:12]:   # top 12 sections
            # Find chunks that overlap with this section
            sec_words = sec.full_text.split()
            sec_tokens = set(w.lower() for w in sec_words[:50])

            # Simple overlap with chunk previews
            relevant = []
            for cs in chunk_scores:
                preview_tokens = set(cs.text_preview.lower().split())
                overlap = len(sec_tokens & preview_tokens)
                if overlap > 3:
                    relevant.append(cs)

            if relevant:
                avg_rec  = float(np.mean([c.recognition  for c in relevant]))
                avg_hier = float(np.mean([c.hierarchy     for c in relevant]))
                avg_conc = float(np.mean([c.concision     for c in relevant]))
            else:
                avg_rec = avg_hier = avg_conc = 0.5

            summary.append({
                "heading":    sec.heading[:80],
                "page":       sec.page_index,
                "words":      sec.word_count,
                "recognition":avg_rec,
                "hierarchy":  avg_hier,
                "concision":  avg_conc,
            })

        return summary
