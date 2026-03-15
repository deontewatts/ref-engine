"""Tests for the scoring engine."""
import tempfile
import os
from ref_engine.file_parser import FileParser
from ref_engine.scoring_engine import REFScoringEngine, RecognitiveScore


def test_score_text_file():
    parser = FileParser()
    engine = REFScoringEngine()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write(
            "This document explores the nature of attention and awareness. "
            "We notice the breath, the body, the present moment. "
            "Together we observe without judgment, grounding in direct "
            "experience rather than abstract theory. The earth beneath "
            "our feet anchors us in physical reality. "
            * 10
        )
        f.flush()
        doc = parser.parse(f.name)
    os.unlink(f.name)

    score = engine.score(doc)
    assert isinstance(score, RecognitiveScore)
    assert score.verdict in (
        "RECOGNITIVE", "PARTIALLY_RECOGNITIVE", "NEUTRAL",
        "HIGH_DISTORTION", "MANIPULATIVE", "EMPTY_DOCUMENT",
    )
    assert 0.0 <= score.overall_score() <= 1.0
    assert len(score.chunk_scores) > 0


def test_empty_document():
    parser = FileParser()
    engine = REFScoringEngine()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write("")
        f.flush()
        doc = parser.parse(f.name)
    os.unlink(f.name)
    score = engine.score(doc)
    assert score.verdict == "EMPTY_DOCUMENT"


def test_validity_flags():
    parser = FileParser()
    engine = REFScoringEngine()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write("Meaningful content. " * 50)
        f.flush()
        doc = parser.parse(f.name)
    os.unlink(f.name)
    score = engine.score(doc)
    flags = score.validity_flags()
    assert "recognition_ok" in flags
    assert "fidelity_ok" in flags
    assert "hierarchy_ok" in flags
    assert "concision_ok" in flags


def test_radar_values():
    parser = FileParser()
    engine = REFScoringEngine()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write("Some interesting content. " * 50)
        f.flush()
        doc = parser.parse(f.name)
    os.unlink(f.name)
    score = engine.score(doc)
    radar = score.radar_values()
    assert "Recognition" in radar
    assert "Fidelity" in radar
    assert "Egalitarian" in radar
    assert "Concision" in radar
    assert "Grounding" in radar
