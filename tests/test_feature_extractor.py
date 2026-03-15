"""Tests for the feature extractor."""
from ref_engine.feature_extractor import TextFeatureExtractor, TextFeatures


def test_extract_returns_features():
    ext = TextFeatureExtractor()
    feat = ext.extract("This is a simple test sentence for extraction.")
    assert isinstance(feat, TextFeatures)
    assert feat.token_count > 0
    assert feat.sentence_count >= 1


def test_extract_empty_text():
    ext = TextFeatureExtractor()
    feat = ext.extract("")
    assert feat.token_count == 0


def test_hierarchy_markers():
    ext = TextFeatureExtractor()
    high_hier = ext.extract(
        "I will teach you the truth. You must understand. "
        "Trust me, I know the answer. Follow me now."
    )
    low_hier = ext.extract(
        "We gather as equals to notice together. "
        "No hierarchy here, anyone can participate freely."
    )
    assert high_hier.hierarchy_score > low_hier.hierarchy_score


def test_concision_grade():
    ext = TextFeatureExtractor()
    feat = ext.extract("Dense. Short. Clear. Facts. Data. Points.")
    assert feat.concision_grade() in ("DENSE", "EFFICIENT", "MODERATE", "DILUTE")


def test_temporal_label():
    ext = TextFeatureExtractor()
    feat = ext.extract("Now here today currently present notice pause feel.")
    assert feat.temporal_label() in (
        "PAST-ORIENTED", "PRESENT-ANCHORED", "FUTURE-ORIENTED"
    )
