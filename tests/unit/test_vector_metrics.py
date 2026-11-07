"""Unit tests for dense vector metrics, normalization, and distance equivalence."""

import math
import numpy as np
import pytest


def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(u * v))


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < eps or norm_v < eps:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.sum((u - v) ** 2)))


def test_cosine_similarity_orthogonal_and_parallel():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    opp = np.array([-1.0, 0.0])

    assert np.isclose(cosine_similarity(u, v), 0.0)
    assert np.isclose(cosine_similarity(u, u), 1.0)
    assert np.isclose(cosine_similarity(u, opp), -1.0)


def test_cosine_scale_invariance():
    u = np.array([3.0, 4.0])
    v = np.array([1.0, 2.0])
    
    sim_original = cosine_similarity(u, v)
    sim_scaled = cosine_similarity(u * 100.0, v * 0.005)
    assert np.isclose(sim_original, sim_scaled)


def test_zero_vector_guard():
    zero = np.zeros(4)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(zero, v) == 0.0


def test_fundamental_equivalence_theorem():
    rng = np.random.RandomState(42)
    for dim in [4, 32, 128, 768]:
        a = rng.randn(dim)
        b = rng.randn(dim)
        
        a_unit = a / np.linalg.norm(a)
        b_unit = b / np.linalg.norm(b)
        
        cos_sim = dot_product(a_unit, b_unit)
        l2_dist = euclidean_distance(a_unit, b_unit)
        
        theoretical_l2 = math.sqrt(max(0.0, 2.0 * (1.0 - cos_sim)))
        assert np.isclose(l2_dist, theoretical_l2, atol=1e-7)
