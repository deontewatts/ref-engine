"""
REF JSON Exporter
-----------------
Serializes a RecognitiveScore to a structured JSON report.
Suitable for downstream pipeline consumption, database storage,
or integration with n8n / Make automations.
"""

import json
import datetime
from ref_engine.scoring_engine import RecognitiveScore


def score_to_dict(score: RecognitiveScore) -> dict:
    """Convert RecognitiveScore to a serializable dict."""
    flags = score.validity_flags()
    radar = score.radar_values()

    return {
        "meta": {
            "framework":   "Recognitive Equation Framework v1.0",
            "timestamp":   datetime.datetime.utcnow().isoformat() + "Z",
            "document":    score.document_name,
        },
        "document_stats": {
            "word_count":     score.word_count,
            "page_count":     score.page_count,
            "equation_count": score.equation_count,
            "heading_count":  score.heading_count,
            "chunk_count":    len(score.chunk_scores),
        },
        "verdict": {
            "label":       score.verdict,
            "code":        score.verdict_code,
            "overall":     round(score.overall_score(), 4),
            "flags":       flags,
        },
        "operator_scores": {
            "recognition":   round(score.mean_recognition,  4),
            "fidelity_dist": round(score.mean_fidelity,      4),
            "hierarchy":     round(score.mean_hierarchy,     4),
            "concision":     round(score.mean_concision,     4),
            "egalitarian":   round(score.mean_egalitarian,   4),
            "affect":        round(score.mean_affect,        4),
            "grounding":     round(score.mean_grounding,     4),
            "temporal_tau":  round(score.mean_temporal,      4),
        },
        "radar": {k: round(v, 4) for k, v in radar.items()},
        "quaternion_state": {
            "initial": {
                "psi0": round(score.initial_psi.components[0], 4),
                "psi1": round(score.initial_psi.components[1], 4),
                "psi2": round(score.initial_psi.components[2], 4),
                "psi3": round(score.initial_psi.components[3], 4),
            } if score.initial_psi else None,
            "final": {
                "psi0": round(score.final_psi.components[0], 4),
                "psi1": round(score.final_psi.components[1], 4),
                "psi2": round(score.final_psi.components[2], 4),
                "psi3": round(score.final_psi.components[3], 4),
            } if score.final_psi else None,
            "displacement": round(score.state_displacement, 4),
        },
        "section_summary": score.section_summary,
        "chunk_detail": [
            {
                "idx":         cs.chunk_index,
                "recognition": round(cs.recognition, 4),
                "fidelity":    round(cs.fidelity,    4),
                "hierarchy":   round(cs.hierarchy,   4),
                "concision":   round(cs.concision,   4),
                "egalitarian": round(cs.egalitarian, 4),
                "temporal":    round(cs.features.temporal_mode, 4),
                "synthetic_p": round(cs.features.synthetic_pressure, 4),
                "preview":     cs.text_preview[:100],
            }
            for cs in score.chunk_scores
        ],
        "comparative_note": _comparative_note(score),
    }


def export_json(score: RecognitiveScore, filepath: str) -> None:
    data = score_to_dict(score)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_jsonl(scores: list, filepath: str) -> None:
    """Export multiple scores as JSONL for batch processing."""
    with open(filepath, "w", encoding="utf-8") as f:
        for score in scores:
            f.write(json.dumps(score_to_dict(score), ensure_ascii=False) + "\n")


def _comparative_note(score: RecognitiveScore) -> str:
    """Generate a one-sentence interpretive note for the score."""
    v = score.verdict
    os_ = score.overall_score()

    if v == "RECOGNITIVE":
        return (f"Document passes all four validity criteria (overall={os_:.3f}). "
                f"It functions as a bounded recognitive operator — moving "
                f"interpreter state toward coherence with minimal distortion.")
    elif v == "MANIPULATIVE":
        return (f"Document fails fidelity or hierarchy thresholds (overall={os_:.3f}). "
                f"High synthetic pressure detected; state is displaced significantly "
                f"from source-node attractor. Treat outputs with caution.")
    elif v == "PARTIALLY_RECOGNITIVE":
        return (f"Document meets core recognition and fidelity criteria but is "
                f"subthreshold on hierarchy or concision (overall={os_:.3f}). "
                f"Structure is coherent; authority framing is present but not dominant.")
    elif v == "HIGH_DISTORTION":
        return (f"Document exceeds distortion or hierarchy thresholds (overall={os_:.3f}). "
                f"Recognitive fidelity is compromised. State drift is above safe bounds.")
    else:
        return (f"Document is indeterminate on recognitive validity (overall={os_:.3f}). "
                f"Insufficient signal or mixed operator effects across chunks.")
