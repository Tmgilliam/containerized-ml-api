"""Shadow mode runner for async model validation."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ShadowConfig:
    """Configuration for shadow mode."""
    enabled: bool = True
    sample_rate: float = 1.0
    max_latency_ms: float = 1000.0
    async_execution: bool = True
    timeout_seconds: float = 5.0
    log_comparisons: bool = True


@dataclass
class ShadowPrediction:
    """Result of a shadow prediction."""
    timestamp: str
    request_id: str
    production_result: dict[str, Any]
    shadow_result: dict[str, Any] | None
    production_latency_ms: float
    shadow_latency_ms: float | None
    shadow_error: str | None = None
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "production_result": self.production_result,
            "shadow_result": self.shadow_result,
            "production_latency_ms": round(self.production_latency_ms, 2),
            "shadow_latency_ms": (
                round(self.shadow_latency_ms, 2) if self.shadow_latency_ms else None
            ),
            "shadow_error": self.shadow_error,
            "comparison": self.comparison,
        }


class ShadowRunner:
    """
    Runs shadow predictions against a candidate model.

    Features:
    - Async execution to not impact production latency
    - Configurable sampling rate
    - Timeout handling
    - Result comparison
    """

    def __init__(
        self,
        production_model: Callable[[dict[str, Any]], dict[str, Any]],
        shadow_model: Callable[[dict[str, Any]], dict[str, Any]],
        config: ShadowConfig | None = None,
        on_shadow_complete: Callable[[ShadowPrediction], None] | None = None,
    ) -> None:
        """
        Initialize shadow runner.

        Args:
            production_model: Function for production predictions
            shadow_model: Function for shadow predictions
            config: Shadow mode configuration
            on_shadow_complete: Callback when shadow prediction completes
        """
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.config = config or ShadowConfig()
        self.on_shadow_complete = on_shadow_complete

        self._executor = ThreadPoolExecutor(max_workers=4)
        self._sample_counter = 0

    def _should_run_shadow(self) -> bool:
        """Determine if shadow should run based on sample rate."""
        if not self.config.enabled:
            return False

        if self.config.sample_rate >= 1.0:
            return True

        import random
        return random.random() < self.config.sample_rate

    def _run_shadow_sync(
        self,
        features: dict[str, Any],
        request_id: str,
        production_result: dict[str, Any],
        production_latency_ms: float,
    ) -> ShadowPrediction:
        """Run shadow prediction synchronously."""
        timestamp = datetime.now(timezone.utc).isoformat()

        shadow_result = None
        shadow_latency_ms = None
        shadow_error = None

        try:
            start = time.perf_counter()
            shadow_result = self.shadow_model(features)
            shadow_latency_ms = (time.perf_counter() - start) * 1000

        except Exception as e:
            shadow_error = str(e)
            logger.warning("Shadow prediction failed: %s", e)

        comparison = self._compare_results(production_result, shadow_result)

        prediction = ShadowPrediction(
            timestamp=timestamp,
            request_id=request_id,
            production_result=production_result,
            shadow_result=shadow_result,
            production_latency_ms=production_latency_ms,
            shadow_latency_ms=shadow_latency_ms,
            shadow_error=shadow_error,
            comparison=comparison,
        )

        if self.on_shadow_complete:
            self.on_shadow_complete(prediction)

        if self.config.log_comparisons:
            logger.info(
                "Shadow comparison: request=%s match=%s prod_risk=%.3f shadow_risk=%.3f",
                request_id,
                comparison.get("predictions_match", False),
                production_result.get("risk_score", 0),
                shadow_result.get("risk_score", 0) if shadow_result else 0,
            )

        return prediction

    def _compare_results(
        self,
        production: dict[str, Any],
        shadow: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compare production and shadow results."""
        if shadow is None:
            return {"shadow_failed": True}

        prod_class = production.get("delay_risk", production.get("prediction"))
        shadow_class = shadow.get("delay_risk", shadow.get("prediction"))

        prod_prob = production.get("risk_score", production.get("probability", 0))
        shadow_prob = shadow.get("risk_score", shadow.get("probability", 0))

        prob_diff = None
        if prod_prob is not None and shadow_prob is not None:
            prob_diff = abs(prod_prob - shadow_prob)

        return {
            "predictions_match": prod_class == shadow_class,
            "production_class": prod_class,
            "shadow_class": shadow_class,
            "production_probability": prod_prob,
            "shadow_probability": shadow_prob,
            "probability_difference": round(prob_diff, 4) if prob_diff is not None else None,
        }

    def predict(
        self,
        features: dict[str, Any],
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], ShadowPrediction | None]:
        """
        Run production prediction with optional shadow.

        Args:
            features: Input features
            request_id: Optional request ID for tracking

        Returns:
            Tuple of (production_result, ShadowPrediction or None)
        """
        request_id = request_id or f"req_{int(time.time() * 1000)}"

        start = time.perf_counter()
        production_result = self.production_model(features)
        production_latency_ms = (time.perf_counter() - start) * 1000

        if not self._should_run_shadow():
            return production_result, None

        if self.config.async_execution:
            self._executor.submit(
                self._run_shadow_sync,
                features,
                request_id,
                production_result,
                production_latency_ms,
            )
            return production_result, None
        else:
            shadow_prediction = self._run_shadow_sync(
                features,
                request_id,
                production_result,
                production_latency_ms,
            )
            return production_result, shadow_prediction

    async def predict_async(
        self,
        features: dict[str, Any],
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], ShadowPrediction | None]:
        """
        Async version of predict.

        Returns production result immediately, shadow runs in background.
        """
        request_id = request_id or f"req_{int(time.time() * 1000)}"

        start = time.perf_counter()
        production_result = self.production_model(features)
        production_latency_ms = (time.perf_counter() - start) * 1000

        if not self._should_run_shadow():
            return production_result, None

        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            self._executor,
            self._run_shadow_sync,
            features,
            request_id,
            production_result,
            production_latency_ms,
        )

        return production_result, None

    def update_shadow_model(
        self,
        new_shadow_model: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Hot-swap the shadow model."""
        self.shadow_model = new_shadow_model
        logger.info("Shadow model updated")

    def enable(self) -> None:
        """Enable shadow mode."""
        self.config.enabled = True
        logger.info("Shadow mode enabled")

    def disable(self) -> None:
        """Disable shadow mode."""
        self.config.enabled = False
        logger.info("Shadow mode disabled")

    def set_sample_rate(self, rate: float) -> None:
        """Set shadow sampling rate (0.0 to 1.0)."""
        self.config.sample_rate = max(0.0, min(1.0, rate))
        logger.info("Shadow sample rate set to %.2f", self.config.sample_rate)

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)
