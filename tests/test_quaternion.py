"""Tests for the quaternion engine."""
import numpy as np
from ref_engine.quaternion import (
    QuaternionState,
    pause_operator,
    notice_operator,
    return_operator,
    recognitive_reset,
)


def test_baseline_is_unit_real():
    s0 = QuaternionState.baseline()
    assert s0.components[0] == 1.0
    assert all(s0.components[1:] == 0.0)
    assert abs(s0.norm() - 1.0) < 1e-9


def test_from_features():
    psi = QuaternionState.from_features(
        attention=0.8, grounding=0.6, memory=0.5, affect=0.3,
    )
    assert psi.components[0] == 0.8
    # imaginary axes mapped from [0,1] to [-1,+1]
    assert abs(psi.components[1] - 0.2) < 1e-9   # 0.6*2-1
    assert abs(psi.components[2] - 0.0) < 1e-9   # 0.5*2-1
    assert abs(psi.components[3] - (-0.4)) < 1e-9  # 0.3*2-1


def test_conjugate():
    psi = QuaternionState(np.array([1.0, 2.0, 3.0, 4.0]))
    conj = psi.conjugate()
    assert conj.components[0] == 1.0
    assert conj.components[1] == -2.0
    assert conj.components[2] == -3.0
    assert conj.components[3] == -4.0


def test_normalize():
    psi = QuaternionState(np.array([3.0, 4.0, 0.0, 0.0]))
    normed = psi.normalize()
    assert abs(normed.norm() - 1.0) < 1e-6


def test_fidelity_identical():
    s0 = QuaternionState.baseline()
    assert abs(s0.fidelity(s0) - 1.0) < 1e-6


def test_fidelity_orthogonal():
    a = QuaternionState(np.array([1.0, 0.0, 0.0, 0.0]))
    b = QuaternionState(np.array([0.0, 1.0, 0.0, 0.0]))
    assert abs(a.fidelity(b)) < 1e-6


def test_hamilton_product():
    a = QuaternionState(np.array([1.0, 0.0, 0.0, 0.0]))
    b = QuaternionState(np.array([0.0, 1.0, 0.0, 0.0]))
    result = a.multiply(b)
    expected = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.components, expected, atol=1e-9)


def test_pause_operator():
    psi = QuaternionState.baseline()
    paused = pause_operator(psi, synthetic_pressure=0.5)
    assert paused.components[0] == 1.0  # real unaffected by gradient
    assert paused.components[1] < psi.components[1]  # dampened


def test_notice_operator():
    psi = QuaternionState(np.array([0.8, 0.3, 0.2, 0.1]))
    noticed = notice_operator(psi)
    assert noticed.components[0] > psi.components[0]


def test_return_operator():
    psi = QuaternionState(np.array([0.5, 0.5, 0.5, 0.5]))
    returned = return_operator(psi)
    s0 = QuaternionState.baseline()
    # Should be closer to baseline after return
    dist_before = np.linalg.norm(psi.components - s0.components)
    dist_after = np.linalg.norm(returned.components - s0.components)
    assert dist_after < dist_before


def test_recognitive_reset():
    psi = QuaternionState(np.array([0.5, 0.5, 0.5, 0.5]))
    reset_psi = recognitive_reset(psi, synthetic_pressure=0.6)
    s0 = QuaternionState.baseline()
    dist_before = np.linalg.norm(psi.components - s0.components)
    dist_after = np.linalg.norm(reset_psi.components - s0.components)
    assert dist_after < dist_before


def test_axis_labels():
    psi = QuaternionState(np.array([0.1, 0.2, 0.3, 0.4]))
    labels = psi.axis_labels()
    assert labels["executive_control"] == 0.1
    assert labels["sensory_grounding"] == 0.2
    assert labels["memory_depth"] == 0.3
    assert labels["affective_valence"] == 0.4
