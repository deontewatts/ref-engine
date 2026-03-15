"""
REF Feature Extractor
----------------------
Translates raw text into the five recognitive operator dimensions:
    𝒮_t  Semantic coherence
    𝒜_t  Attentional stability
    𝒞_t  Concision density  C(T) = I(T) / L(T)
    ℰ_t  Affective resonance
    𝒬_t  Temporal-branching mode τ ∈ {−1, 0, +1}

Plus hierarchy functional ℋ(T) — the penalty for asymmetric status encoding.

All computed without heavy ML models, using:
    - Token entropy (Shannon H over word frequencies)
    - Syntactic pattern matching (regex + heuristics)
    - Lexical affect dictionaries (embedded inline)
    - Structural analysis (sentence length variance, question density)
"""

import re
import math
import collections
import numpy as np
from typing import List
from dataclasses import dataclass


# ─── Embedded Lexicons ────────────────────────────────────────────────────────

# High-affect words (arousal dimension) — presence raises ℰ_t
HIGH_AFFECT_WORDS = {
    "critical", "crucial", "urgent", "vital", "danger", "threat", "collapse",
    "sovereign", "ultimate", "absolute", "catastrophic", "lethal", "terror",
    "fear", "anger", "grief", "rage", "despair", "love", "joy", "awe",
    "wonder", "beauty", "sacred", "divine", "eternal", "infinite", "death",
    "birth", "war", "freedom", "prison", "escape", "liberation", "awaken",
    "truth", "lie", "deceive", "betray", "trust", "faith", "doubt",
    "power", "control", "dominate", "surrender", "resistance", "rebel",
}

# Grounding words — raise ψ₁ (sensory/concrete axis)
GROUNDING_WORDS = {
    "body", "breath", "feet", "hands", "skin", "touch", "sound", "light",
    "water", "earth", "stone", "tree", "air", "weight", "presence", "now",
    "here", "physical", "concrete", "material", "real", "actual", "direct",
    "notice", "feel", "sense", "observe", "watch", "listen", "ground",
    "pause", "still", "quiet", "rest", "sit", "stand", "walk", "move",
}

# Hierarchy markers — raise ℋ(T), penalize E_g
HIERARCHY_MARKERS = {
    # Epistemic ownership: "I will teach you", "you must understand"
    "teach you", "you must", "you should", "you need to", "you have to",
    "listen to me", "trust me", "i know", "i will show", "follow me",
    "my teaching", "my truth", "the answer is", "the truth is",
    # Authority/credential claims
    "as an expert", "as a master", "proven method", "only way",
    "guaranteed", "scientifically proven", "ancient wisdom says",
    # Asymmetric address
    "disciples", "followers", "students", "seekers", "those who",
}

# Egalitarian/non-hierarchy markers — reduce ℋ(T)
EGALITARIAN_MARKERS = {
    "we gather", "as equals", "together", "shared", "notice together",
    "no hierarchy", "no masters", "no followers", "horizontal",
    "no one owns", "no one discovered", "we can", "each of us",
    "anyone can", "open source", "freely", "without permission",
}

# Temporal mode markers
PAST_MARKERS = {"was", "were", "had", "did", "ago", "before", "previously",
                "history", "ancient", "once", "used to", "formerly"}
FUTURE_MARKERS = {"will", "shall", "going to", "plan to", "intend", "predict",
                  "forecast", "eventually", "soon", "future", "tomorrow",
                  "upcoming", "next", "hope to", "expect"}
PRESENT_MARKERS = {"now", "is", "are", "here", "today", "currently",
                   "at this moment", "present", "notice", "pause", "feel"}

# Imperative / command starters (raises ℋ if subject-asymmetric)
IMPERATIVE_STARTERS = re.compile(
    r"^\s*(must|should|need to|have to|do not|don't|stop|start|always|never)\b",
    re.IGNORECASE
)


# ─── Core Extractor ───────────────────────────────────────────────────────────

@dataclass
class TextFeatures:
    """Complete feature vector for a text chunk."""
    # Quaternion axis estimates [0,1]
    attention_stability: float    # ψ₀ proxy
    sensory_grounding:   float    # ψ₁ proxy
    memory_depth:        float    # ψ₂ proxy (lexical richness, reference density)
    affective_valence:   float    # ψ₃ proxy

    # Recognitive operator scores
    semantic_coherence:  float    # 𝒮: local semantic consistency
    attentional_load:    float    # 𝒜: how much competing attention is demanded
    concision_density:   float    # 𝒞: I(T)/L(T) — information per token
    affective_resonance: float    # ℰ: emotional engagement
    temporal_mode:       float    # 𝒬: τ ∈ [−1, +1]  (past↔future, 0=present)

    # Hierarchy functional
    hierarchy_score:     float    # ℋ(T) ∈ [0,1]  (0=fully egalitarian)

    # Metadata
    token_count:         int
    sentence_count:      int
    entropy:             float    # Shannon entropy of word distribution
    question_density:    float    # fraction of sentences that are questions
    synthetic_pressure:  float    # estimated external manipulation signal

    def egalitarian_score(self, alpha: float = 0.4) -> float:
        """E_g(T) = Rec − α·ℋ(T)  — penalise hierarchy in recognition score."""
        return max(0.0, self.semantic_coherence - alpha * self.hierarchy_score)

    def concision_grade(self) -> str:
        c = self.concision_density
        if c > 0.75:   return "DENSE"
        if c > 0.50:   return "EFFICIENT"
        if c > 0.30:   return "MODERATE"
        return "DILUTE"

    def temporal_label(self) -> str:
        t = self.temporal_mode
        if t < -0.3:   return "PAST-ORIENTED"
        if t > 0.3:    return "FUTURE-ORIENTED"
        return "PRESENT-ANCHORED"


class TextFeatureExtractor:
    """
    Agentically processes a text string and returns a TextFeatures instance.
    All scoring is deterministic and interpretable — no black-box ML.
    """

    def extract(self, text: str) -> TextFeatures:
        tokens      = self._tokenize(text)
        sentences   = self._split_sentences(text)
        words       = [t.lower() for t in tokens if t.isalpha()]

        if not words:
            return self._empty_features()

        freq        = collections.Counter(words)
        entropy     = self._shannon_entropy(freq, len(words))
        vocab_size  = len(freq)
        n_tokens    = len(words)
        n_sentences = max(1, len(sentences))

        # ── Axis estimates ────────────────────────────────────────────────────
        # ψ₀ Executive Control / Attention Stability
        # Sentence length variance: high variance → fragmented attention
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sent_lengths) > 1:
            length_cv = (np.std(sent_lengths) / (np.mean(sent_lengths) + 1e-9))
        else:
            length_cv = 0.0
        attention_stability = max(0.0, 1.0 - min(1.0, length_cv / 2.0))

        # ψ₁ Sensory Grounding
        grounding_hits = sum(1 for w in words if w in GROUNDING_WORDS)
        sensory_grounding = min(1.0, grounding_hits / (n_tokens * 0.05 + 1))

        # ψ₂ Memory Depth — type-token ratio (lexical richness) × reference density
        ttr = vocab_size / (n_tokens + 1e-9)
        ref_density = self._reference_density(text)
        memory_depth = min(1.0, (ttr + ref_density) / 2.0)

        # ψ₃ Affective Valence
        affect_hits = sum(1 for w in words if w in HIGH_AFFECT_WORDS)
        affective_valence = min(1.0, affect_hits / (n_tokens * 0.08 + 1))

        # ── Operator scores ───────────────────────────────────────────────────
        semantic_coherence  = self._semantic_coherence(words, freq, n_tokens)
        attentional_load    = 1.0 - attention_stability
        concision_density   = self._concision_density(entropy, n_tokens,
                                                       n_sentences)
        affective_resonance = affective_valence
        temporal_mode       = self._temporal_mode(words)

        # ── Hierarchy ─────────────────────────────────────────────────────────
        hierarchy_score = self._hierarchy_score(text, words, sentences)

        # ── Synthetic pressure ────────────────────────────────────────────────
        # Proxy: high affect + high hierarchy + low grounding = manipulation risk
        synthetic_pressure = min(1.0,
            0.4 * hierarchy_score +
            0.3 * affective_valence +
            0.3 * (1.0 - sensory_grounding)
        )

        question_density = sum(1 for s in sentences
                               if s.strip().endswith("?")) / n_sentences

        return TextFeatures(
            attention_stability = attention_stability,
            sensory_grounding   = sensory_grounding,
            memory_depth        = memory_depth,
            affective_valence   = affective_valence,
            semantic_coherence  = semantic_coherence,
            attentional_load    = attentional_load,
            concision_density   = concision_density,
            affective_resonance = affective_resonance,
            temporal_mode       = temporal_mode,
            hierarchy_score     = hierarchy_score,
            token_count         = n_tokens,
            sentence_count      = n_sentences,
            entropy             = entropy,
            question_density    = question_density,
            synthetic_pressure  = synthetic_pressure,
        )

    # ─── Private methods ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text)

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", text.strip())

    def _shannon_entropy(self, freq: collections.Counter, total: int) -> float:
        if total == 0:
            return 0.0
        h = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def _reference_density(self, text: str) -> float:
        """Estimate depth by counting citations, parenthetical refs, equations."""
        patterns = [
            r"\([A-Z][a-z]+,?\s+\d{4}\)",   # (Author, Year)
            r"\[\d+\]",                        # [1]
            r"[A-Z]\s*=\s*[A-Za-z₀-₉+\-*/]",  # equations
            r"\b(theorem|lemma|proof|axiom|definition|proposition)\b",
            r"\b(equation|formula|matrix|vector|tensor|operator)\b",
            r"\b(section|chapter|figure|table)\s+\d",
        ]
        hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
        text_len = max(1, len(text.split()))
        return min(1.0, hits / (text_len * 0.02 + 1))

    def _semantic_coherence(self, words: List[str],
                            freq: collections.Counter, n: int) -> float:
        """
        Proxy for local semantic coherence using bigram repetition.
        Higher repetition of content bigrams → more coherent, focused text.
        Also rewards presence of connective tissue words.
        """
        if n < 4:
            return 0.5
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_freq = collections.Counter(bigrams)
        repeated = sum(1 for c in bigram_freq.values() if c > 1)
        coherence = min(1.0, repeated / (len(bigrams) * 0.1 + 1))

        # Reward connective tissue
        connectors = {"therefore", "because", "thus", "hence", "however",
                      "although", "whereas", "which", "that", "since",
                      "when", "as", "so", "but", "yet", "and"}
        conn_hits = sum(1 for w in words if w in connectors)
        conn_score = min(1.0, conn_hits / (n * 0.05 + 1))

        return 0.6 * coherence + 0.4 * conn_score

    def _concision_density(self, entropy: float,
                           n_tokens: int, n_sentences: int) -> float:
        """
        C(T) = I(T) / L(T)
        Approximate I(T) = entropy × log(vocab) / max_entropy
        Normalized by token count so shorter, denser texts score higher.
        """
        # Max entropy for this token count (uniform distribution)
        max_entropy = math.log2(n_tokens) if n_tokens > 1 else 1.0
        information_ratio = entropy / (max_entropy + 1e-9)

        # Penalize average sentence length above ~15 words
        avg_len = n_tokens / (n_sentences + 1e-9)
        length_penalty = max(0.0, 1.0 - (avg_len - 15) / 30.0)

        return float(np.clip(information_ratio * length_penalty, 0.0, 1.0))

    def _temporal_mode(self, words: List[str]) -> float:
        """
        τ ∈ [−1, +1]
        −1 = strongly past-anchored
        0  = present-centered
        +1 = strongly future-oriented
        """
        past_hits    = sum(1 for w in words if w in PAST_MARKERS)
        future_hits  = sum(1 for w in words if w in FUTURE_MARKERS)
        present_hits = sum(1 for w in words if w in PRESENT_MARKERS)

        total = past_hits + future_hits + present_hits + 1e-9
        tau = (future_hits - past_hits) / total
        # If present is dominant, pull toward zero
        if present_hits > past_hits + future_hits:
            tau *= 0.3
        return float(np.clip(tau, -1.0, 1.0))

    def _hierarchy_score(self, text: str,
                         words: List[str], sentences: List[str]) -> float:
        """
        ℋ(T) ∈ [0,1]
        Penalizes:
          • Asymmetric epistemic claims ("I will teach you")
          • High imperative density
          • Credential assertion
          • Low egalitarian marker presence
        """
        text_lower = text.lower()

        # Hierarchy marker hits
        h_hits = sum(1 for m in HIERARCHY_MARKERS if m in text_lower)
        h_score = min(1.0, h_hits / 5.0)

        # Egalitarian marker hits (reduces hierarchy)
        e_hits = sum(1 for m in EGALITARIAN_MARKERS if m in text_lower)
        e_score = min(1.0, e_hits / 4.0)

        # Imperative density
        imp_count = sum(1 for s in sentences
                        if IMPERATIVE_STARTERS.match(s.strip()))
        imp_density = imp_count / (len(sentences) + 1e-9)

        # Pronoun asymmetry: heavy 1st-person vs 2nd-person
        i_count  = sum(1 for w in words if w in {"i", "my", "me", "mine"})
        you_count = sum(1 for w in words if w in {"you", "your", "yours"})
        asym = 0.0
        if (i_count + you_count) > 0:
            asym = abs(i_count - you_count) / (i_count + you_count)

        raw = 0.35 * h_score + 0.25 * imp_density + 0.25 * asym - 0.15 * e_score
        return float(np.clip(raw, 0.0, 1.0))

    def _empty_features(self) -> TextFeatures:
        return TextFeatures(
            attention_stability=0.5, sensory_grounding=0.5,
            memory_depth=0.5, affective_valence=0.5,
            semantic_coherence=0.5, attentional_load=0.5,
            concision_density=0.5, affective_resonance=0.5,
            temporal_mode=0.0, hierarchy_score=0.5,
            token_count=0, sentence_count=0, entropy=0.0,
            question_density=0.0, synthetic_pressure=0.5,
        )
