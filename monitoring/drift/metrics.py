"""Statistical metrics for drift detection."""

from __future__ import annotations

import numpy as np
from scipy import stats


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI measures the shift in distribution between a reference (expected) and
    current (actual) dataset. Used to detect feature or prediction drift.
    
    Interpretation:
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change, monitor closely
        PSI >= 0.2: Significant change, investigate
    
    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (production data)
        buckets: Number of bins for histogram
        epsilon: Small value to avoid division by zero
    
    Returns:
        PSI score (non-negative float)
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def calculate_ks_statistic(
    expected: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic between two distributions.
    
    KS test measures the maximum distance between cumulative distributions.
    Useful for detecting subtle distribution shifts.
    
    Args:
        expected: Reference distribution
        actual: Current distribution
    
    Returns:
        Tuple of (KS statistic, p-value)
    """
    statistic, pvalue = stats.ks_2samp(expected, actual)
    return float(statistic), float(pvalue)


def calculate_chi_square(
    expected: np.ndarray,
    actual: np.ndarray,
    categories: list[str] | None = None,
) -> tuple[float, float]:
    """
    Calculate Chi-Square statistic for categorical drift detection.
    
    Args:
        expected: Reference categorical distribution (counts or raw values)
        actual: Current categorical distribution
        categories: Optional list of category names for alignment
    
    Returns:
        Tuple of (chi-square statistic, p-value)
    """
    if categories is None:
        categories = sorted(set(expected) | set(actual))
    
    expected_counts = np.array([np.sum(expected == cat) for cat in categories])
    actual_counts = np.array([np.sum(actual == cat) for cat in categories])
    
    expected_counts = expected_counts + 1
    actual_counts = actual_counts + 1
    
    expected_freq = expected_counts / expected_counts.sum() * actual_counts.sum()
    
    chi2, pvalue = stats.chisquare(actual_counts, f_exp=expected_freq)
    return float(chi2), float(pvalue)


def calculate_jensen_shannon_divergence(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Calculate Jensen-Shannon Divergence between two distributions.
    
    JSD is a symmetric, bounded (0-1) measure of distribution similarity.
    
    Args:
        expected: Reference distribution
        actual: Current distribution
        buckets: Number of bins for histogram
    
    Returns:
        JSD score (0 = identical, 1 = maximally different)
    """
    breakpoints = np.percentile(
        np.concatenate([expected, actual]), 
        np.linspace(0, 100, buckets + 1)
    )
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    p = np.histogram(expected, bins=breakpoints)[0].astype(float)
    q = np.histogram(actual, bins=breakpoints)[0].astype(float)
    
    p = p / p.sum() + 1e-10
    q = q / q.sum() + 1e-10
    
    m = 0.5 * (p + q)
    
    jsd = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
    return float(jsd)
