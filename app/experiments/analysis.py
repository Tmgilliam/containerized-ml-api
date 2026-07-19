"""Statistical analysis for A/B experiments."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results of statistical analysis for an experiment."""
    experiment_name: str
    control_variant: str
    treatment_variant: str
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int
    absolute_difference: float
    relative_difference: float
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    statistical_power: float
    minimum_detectable_effect: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "control_variant": self.control_variant,
            "treatment_variant": self.treatment_variant,
            "metric_name": self.metric_name,
            "control": {
                "mean": round(self.control_mean, 4),
                "std": round(self.control_std, 4),
                "n": self.control_n,
            },
            "treatment": {
                "mean": round(self.treatment_mean, 4),
                "std": round(self.treatment_std, 4),
                "n": self.treatment_n,
            },
            "absolute_difference": round(self.absolute_difference, 4),
            "relative_difference_pct": round(self.relative_difference * 100, 2),
            "p_value": round(self.p_value, 4),
            "confidence_interval_95": [round(x, 4) for x in self.confidence_interval],
            "is_significant": self.is_significant,
            "statistical_power": round(self.statistical_power, 2),
            "minimum_detectable_effect": round(self.minimum_detectable_effect, 4),
        }


class ExperimentAnalyzer:
    """
    Statistical analysis engine for A/B experiments.
    
    Provides:
    - Two-sample t-tests for continuous metrics
    - Chi-squared tests for conversion rates
    - Power analysis and sample size calculations
    - Multiple comparison corrections
    """
    
    DEFAULT_ALPHA = 0.05
    DEFAULT_POWER = 0.8
    
    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        min_samples_per_variant: int = 100,
    ) -> None:
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level (default 0.05)
            min_samples_per_variant: Minimum samples needed for valid analysis
        """
        self.alpha = alpha
        self.min_samples_per_variant = min_samples_per_variant
    
    def _calculate_power(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        pooled_std: float,
    ) -> float:
        """Calculate statistical power of the test."""
        if pooled_std == 0:
            return 1.0
        
        standardized_effect = effect_size / pooled_std
        se = pooled_std * math.sqrt(1/n1 + 1/n2)
        
        if se == 0:
            return 1.0
        
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_stat = abs(effect_size) / se
        
        power = 1 - stats.norm.cdf(z_alpha - z_stat) + stats.norm.cdf(-z_alpha - z_stat)
        return min(1.0, max(0.0, power))
    
    def _calculate_mde(
        self,
        n1: int,
        n2: int,
        pooled_std: float,
        power: float = DEFAULT_POWER,
    ) -> float:
        """Calculate minimum detectable effect size."""
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_power = stats.norm.ppf(power)
        
        se = pooled_std * math.sqrt(1/n1 + 1/n2)
        mde = (z_alpha + z_power) * se
        
        return mde
    
    def analyze_continuous_metric(
        self,
        experiment_name: str,
        control_variant: str,
        treatment_variant: str,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        metric_name: str = "metric",
    ) -> ExperimentResult:
        """
        Analyze a continuous metric using Welch's t-test.
        
        Args:
            experiment_name: Name of the experiment
            control_variant: Name of control variant
            treatment_variant: Name of treatment variant
            control_values: Array of metric values for control
            treatment_values: Array of metric values for treatment
            metric_name: Name of the metric being analyzed
        
        Returns:
            ExperimentResult with statistical analysis
        """
        n1, n2 = len(control_values), len(treatment_values)
        
        if n1 < self.min_samples_per_variant or n2 < self.min_samples_per_variant:
            logger.warning(
                "Insufficient samples for %s: control=%d, treatment=%d (min=%d)",
                experiment_name, n1, n2, self.min_samples_per_variant,
            )
        
        control_mean = float(np.mean(control_values))
        treatment_mean = float(np.mean(treatment_values))
        control_std = float(np.std(control_values, ddof=1)) if n1 > 1 else 0.0
        treatment_std = float(np.std(treatment_values, ddof=1)) if n2 > 1 else 0.0
        
        statistic, p_value = stats.ttest_ind(
            control_values,
            treatment_values,
            equal_var=False,
        )
        
        pooled_std = math.sqrt(
            ((n1 - 1) * control_std**2 + (n2 - 1) * treatment_std**2) / (n1 + n2 - 2)
        ) if n1 + n2 > 2 else max(control_std, treatment_std)
        
        se = math.sqrt(control_std**2/n1 + treatment_std**2/n2) if n1 > 0 and n2 > 0 else 0
        diff = treatment_mean - control_mean
        t_crit = stats.t.ppf(1 - self.alpha/2, min(n1, n2) - 1) if min(n1, n2) > 1 else 1.96
        ci = (diff - t_crit * se, diff + t_crit * se)
        
        relative_diff = diff / control_mean if control_mean != 0 else 0
        
        power = self._calculate_power(abs(diff), n1, n2, pooled_std)
        mde = self._calculate_mde(n1, n2, pooled_std)
        
        return ExperimentResult(
            experiment_name=experiment_name,
            control_variant=control_variant,
            treatment_variant=treatment_variant,
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_std=control_std,
            treatment_std=treatment_std,
            control_n=n1,
            treatment_n=n2,
            absolute_difference=diff,
            relative_difference=relative_diff,
            p_value=float(p_value),
            confidence_interval=ci,
            is_significant=p_value < self.alpha,
            statistical_power=power,
            minimum_detectable_effect=mde,
        )
    
    def analyze_conversion_rate(
        self,
        experiment_name: str,
        control_variant: str,
        treatment_variant: str,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        metric_name: str = "conversion_rate",
    ) -> ExperimentResult:
        """
        Analyze a conversion rate metric using chi-squared test.
        
        Args:
            experiment_name: Name of the experiment
            control_variant: Name of control variant
            treatment_variant: Name of treatment variant
            control_conversions: Number of conversions in control
            control_total: Total samples in control
            treatment_conversions: Number of conversions in treatment
            treatment_total: Total samples in treatment
            metric_name: Name of the metric
        
        Returns:
            ExperimentResult with statistical analysis
        """
        contingency = np.array([
            [control_conversions, control_total - control_conversions],
            [treatment_conversions, treatment_total - treatment_conversions],
        ])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        control_rate = control_conversions / control_total if control_total > 0 else 0
        treatment_rate = treatment_conversions / treatment_total if treatment_total > 0 else 0
        
        control_std = math.sqrt(control_rate * (1 - control_rate)) if 0 < control_rate < 1 else 0
        treatment_std = math.sqrt(treatment_rate * (1 - treatment_rate)) if 0 < treatment_rate < 1 else 0
        
        diff = treatment_rate - control_rate
        se = math.sqrt(
            control_rate * (1 - control_rate) / control_total +
            treatment_rate * (1 - treatment_rate) / treatment_total
        ) if control_total > 0 and treatment_total > 0 else 0
        
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        ci = (diff - z_crit * se, diff + z_crit * se)
        
        relative_diff = diff / control_rate if control_rate != 0 else 0
        
        pooled_rate = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        pooled_std = math.sqrt(pooled_rate * (1 - pooled_rate)) if 0 < pooled_rate < 1 else 0
        
        power = self._calculate_power(abs(diff), control_total, treatment_total, pooled_std)
        mde = self._calculate_mde(control_total, treatment_total, pooled_std)
        
        return ExperimentResult(
            experiment_name=experiment_name,
            control_variant=control_variant,
            treatment_variant=treatment_variant,
            metric_name=metric_name,
            control_mean=control_rate,
            treatment_mean=treatment_rate,
            control_std=control_std,
            treatment_std=treatment_std,
            control_n=control_total,
            treatment_n=treatment_total,
            absolute_difference=diff,
            relative_difference=relative_diff,
            p_value=float(p_value),
            confidence_interval=ci,
            is_significant=p_value < self.alpha,
            statistical_power=power,
            minimum_detectable_effect=mde,
        )
    
    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = DEFAULT_POWER,
    ) -> int:
        """
        Calculate required sample size per variant.
        
        Args:
            baseline_rate: Expected baseline conversion/metric rate
            minimum_detectable_effect: Smallest effect you want to detect
            power: Desired statistical power (default 0.8)
        
        Returns:
            Required sample size per variant
        """
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_power = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        
        pooled_p = (p1 + p2) / 2
        pooled_var = pooled_p * (1 - pooled_p)
        
        if pooled_var == 0:
            return self.min_samples_per_variant
        
        n = 2 * pooled_var * (z_alpha + z_power) ** 2 / minimum_detectable_effect ** 2
        
        return max(self.min_samples_per_variant, int(math.ceil(n)))
    
    def analyze_experiment(
        self,
        experiment_name: str,
        data: dict[str, list[dict[str, Any]]],
        control_variant: str,
        metrics: list[str] | None = None,
    ) -> list[ExperimentResult]:
        """
        Analyze all metrics for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            data: Dict mapping variant names to list of records
            control_variant: Name of the control variant
            metrics: List of metric names to analyze (default: risk_score, latency_ms)
        
        Returns:
            List of ExperimentResult for each metric and treatment variant
        """
        if metrics is None:
            metrics = ["risk_score", "latency_ms"]
        
        if control_variant not in data:
            raise ValueError(f"Control variant '{control_variant}' not in data")
        
        control_data = data[control_variant]
        results = []
        
        for variant_name, variant_data in data.items():
            if variant_name == control_variant:
                continue
            
            for metric in metrics:
                control_values = np.array([
                    r[metric] for r in control_data
                    if r.get(metric) is not None
                ])
                treatment_values = np.array([
                    r[metric] for r in variant_data
                    if r.get(metric) is not None
                ])
                
                if len(control_values) == 0 or len(treatment_values) == 0:
                    logger.warning(
                        "No data for metric %s in experiment %s",
                        metric, experiment_name,
                    )
                    continue
                
                result = self.analyze_continuous_metric(
                    experiment_name=experiment_name,
                    control_variant=control_variant,
                    treatment_variant=variant_name,
                    control_values=control_values,
                    treatment_values=treatment_values,
                    metric_name=metric,
                )
                results.append(result)
            
            control_outcomes = [r for r in control_data if r.get("outcome") is not None]
            treatment_outcomes = [r for r in variant_data if r.get("outcome") is not None]
            
            if control_outcomes and treatment_outcomes:
                result = self.analyze_conversion_rate(
                    experiment_name=experiment_name,
                    control_variant=control_variant,
                    treatment_variant=variant_name,
                    control_conversions=sum(1 for r in control_outcomes if r["outcome"]),
                    control_total=len(control_outcomes),
                    treatment_conversions=sum(1 for r in treatment_outcomes if r["outcome"]),
                    treatment_total=len(treatment_outcomes),
                    metric_name="actual_delay_rate",
                )
                results.append(result)
        
        return results
