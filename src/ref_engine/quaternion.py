"""
REF Quaternion Engine
---------------------
Implements the four-dimensional cognitive state vector:
    Ψ = ψ₀ + ψ₁i + ψ₂j + ψ₃k
where axes map to:
    ψ₀  Executive Control   (attention / inhibitory focus)
    ψ₁  Sensory Input       (grounding / concrete reality signal)
    ψ₂  Memory Retrieval    (contextual depth / prior knowledge load)
    ψ₃  Emotional Valence   (affective resonance / arousal)

All operations preserve the Hamilton algebra: ij=k, jk=i, ki=j, i²=j²=k²=-1
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class QuaternionState:
    """
    A single cognitive state Ψ ∈ ℍ^d.
    We store as a (4,) numpy array [ψ₀, ψ₁, ψ₂, ψ₃].
    Normalized to unit quaternion before any fidelity computation.
    """
    components: np.ndarray  # shape (4,)

    def __post_init__(self):
        self.components = np.array(self.components, dtype=np.float64)
        if self.components.shape != (4,):
            raise ValueError("QuaternionState requires exactly 4 components")

    @classmethod
    def baseline(cls) -> "QuaternionState":
        """
        Source-node state S₀ = [1, 0, 0, 0].
        Pure real quaternion — full executive control, no distortion.
        This is the 'Return' target in the Pause-Notice-Return cycle.
        """
        return cls(np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_features(cls, attention: float, grounding: float,
                      memory: float, affect: float) -> "QuaternionState":
        """Construct state from normalized [0,1] feature scores."""
        v = np.array([attention, grounding, memory, affect], dtype=np.float64)
        # Map [0,1] → [−1,+1] for ψ₁ψ₂ψ₃ (imaginary axes can be negative)
        v[1:] = v[1:] * 2.0 - 1.0
        return cls(v)

    def norm(self) -> float:
        return float(np.linalg.norm(self.components))

    def normalize(self) -> "QuaternionState":
        n = self.norm()
        return QuaternionState(self.components / (n + 1e-9))

    def conjugate(self) -> "QuaternionState":
        """Ψ* = [ψ₀, −ψ₁, −ψ₂, −ψ₃]"""
        c = self.components.copy()
        c[1:] = -c[1:]
        return QuaternionState(c)

    def multiply(self, other: "QuaternionState") -> "QuaternionState":
        """
        Hamilton product Ψ₁ × Ψ₂.
        Non-commutative — critical for modeling operator ordering effects.
        """
        a0, a1, a2, a3 = self.components
        b0, b1, b2, b3 = other.components
        return QuaternionState(np.array([
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0,
        ]))

    def inner_product(self, other: "QuaternionState") -> float:
        """Standard R⁴ inner product ⟨Ψ, Φ⟩ = Ψ·Φ"""
        return float(np.dot(self.components, other.components))

    def angular_distance(self, other: "QuaternionState") -> float:
        """
        Geodesic distance on unit 3-sphere.
        Returns angle in radians ∈ [0, π].
        """
        dot = np.clip(self.normalize().inner_product(other.normalize()), -1.0, 1.0)
        return float(np.arccos(abs(dot)))

    def fidelity(self, other: "QuaternionState") -> float:
        """
        Recognitive fidelity ℰ_F ∈ [0,1].
        1.0 = states are identical (perfect fidelity)
        0.0 = states are orthogonal (complete distortion)
        """
        n1 = self.normalize()
        n2 = other.normalize()
        return float(abs(np.dot(n1.components, n2.components)))

    def axis_labels(self) -> dict:
        return {
            "executive_control": float(self.components[0]),
            "sensory_grounding": float(self.components[1]),
            "memory_depth":      float(self.components[2]),
            "affective_valence": float(self.components[3]),
        }

    def __repr__(self) -> str:
        c = self.components
        return (f"Ψ = {c[0]:+.3f} + {c[1]:+.3f}i "
                f"+ {c[2]:+.3f}j + {c[3]:+.3f}k  |Ψ|={self.norm():.3f}")


def pause_operator(psi: QuaternionState, synthetic_pressure: float,
                   lam: float = 0.3) -> QuaternionState:
    """
    𝒫(Ψ) = Ψ − λ·p·∇_Ξ Ψ
    Dampen synthetic-field coupling.
    synthetic_pressure ∈ [0,1]: how much external algorithmic forcing exists.
    lam: coupling constant (default 0.3, tunable).
    """
    gradient = np.array([0, synthetic_pressure, synthetic_pressure,
                         synthetic_pressure * 0.5])
    new_components = psi.components - lam * gradient
    return QuaternionState(np.clip(new_components, -2.0, 2.0))


def notice_operator(psi: QuaternionState, mu: float = 0.2) -> QuaternionState:
    """
    𝒩(Ψ) = Ψ + μ·∇_Ψ|Ψ|²
    Increase self-observability — amplify the real (executive) component
    proportional to current norm, which represents meta-cognitive awareness.
    """
    grad_norm_sq = 2.0 * psi.components
    boost = np.zeros(4)
    boost[0] = mu * grad_norm_sq[0]   # strengthen executive control
    return QuaternionState(psi.components + boost)


def return_operator(psi: QuaternionState,
                    source: QuaternionState = None) -> QuaternionState:
    """
    ℛ(Ψ) = Π₀(Ψ): project toward source-node attractor S₀.
    Interpolates 30% back toward S₀ without full collapse —
    recognition, not erasure.
    """
    if source is None:
        source = QuaternionState.baseline()
    # Slerp-style blend: 70% current + 30% source
    blended = 0.7 * psi.components + 0.3 * source.components
    return QuaternionState(blended)


def recognitive_reset(psi: QuaternionState,
                      synthetic_pressure: float) -> QuaternionState:
    """
    Full Pause→Notice→Return cycle:
    Ψ_{k+1} = ℛ(𝒩(𝒫(Ψ_k)))
    This is the minimal recognitive reset law from the framework.
    """
    psi = pause_operator(psi, synthetic_pressure)
    psi = notice_operator(psi)
    psi = return_operator(psi)
    return psi
