"""Alert notification system for drift and model issues."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from monitoring.drift.detector import DriftResult, DriftSeverity

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    """Supported alert channels."""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class AlertPayload:
    """Standardized alert payload."""
    title: str
    message: str
    severity: str
    timestamp: str
    model_version: str
    details: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "details": self.details,
        }


class BaseNotifier(ABC):
    """Base class for alert notifiers."""
    
    @abstractmethod
    def send(self, payload: AlertPayload) -> bool:
        """Send alert notification. Returns True if successful."""
        pass


class SlackNotifier(BaseNotifier):
    """Send alerts to Slack via webhook."""
    
    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        
    def _format_slack_message(self, payload: AlertPayload) -> dict[str, Any]:
        """Format payload as Slack block kit message."""
        severity_emoji = {
            "none": ":white_check_mark:",
            "low": ":warning:",
            "moderate": ":large_orange_diamond:",
            "high": ":red_circle:",
            "critical": ":rotating_light:",
        }
        
        emoji = severity_emoji.get(payload.severity, ":question:")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {payload.title}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": payload.message,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity:*\n{payload.severity.upper()}"},
                    {"type": "mrkdwn", "text": f"*Model:*\n{payload.model_version}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{payload.timestamp}"},
                ],
            },
        ]
        
        if payload.details:
            details_text = "\n".join(f"• {k}: {v}" for k, v in payload.details.items())
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{details_text}",
                },
            })
        
        return {"blocks": blocks}
    
    def send(self, payload: AlertPayload) -> bool:
        if not self.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False
        
        try:
            slack_payload = self._format_slack_message(payload)
            data = json.dumps(slack_payload).encode("utf-8")
            
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info("Slack alert sent successfully")
                    return True
                    
        except URLError as e:
            logger.error("Failed to send Slack alert: %s", e)
        except Exception as e:
            logger.error("Unexpected error sending Slack alert: %s", e)
        
        return False


class PagerDutyNotifier(BaseNotifier):
    """Send alerts to PagerDuty via Events API v2."""
    
    EVENTS_API_URL = "https://events.pagerduty.com/v2/enqueue"
    
    def __init__(
        self,
        routing_key: str | None = None,
        service_name: str = "ML Model Monitoring",
    ) -> None:
        self.routing_key = routing_key or os.getenv("PAGERDUTY_ROUTING_KEY")
        self.service_name = service_name
        
    def _severity_to_pd(self, severity: str) -> str:
        """Map internal severity to PagerDuty severity."""
        mapping = {
            "none": "info",
            "low": "warning",
            "moderate": "warning",
            "high": "error",
            "critical": "critical",
        }
        return mapping.get(severity, "info")
    
    def send(self, payload: AlertPayload) -> bool:
        if not self.routing_key:
            logger.error("PagerDuty routing key not configured")
            return False
        
        try:
            pd_payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": f"{payload.model_version}-{payload.severity}",
                "payload": {
                    "summary": payload.title,
                    "source": self.service_name,
                    "severity": self._severity_to_pd(payload.severity),
                    "timestamp": payload.timestamp,
                    "custom_details": {
                        "message": payload.message,
                        "model_version": payload.model_version,
                        **payload.details,
                    },
                },
            }
            
            data = json.dumps(pd_payload).encode("utf-8")
            req = Request(
                self.EVENTS_API_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urlopen(req, timeout=10) as response:
                if response.status == 202:
                    logger.info("PagerDuty alert sent successfully")
                    return True
                    
        except URLError as e:
            logger.error("Failed to send PagerDuty alert: %s", e)
        except Exception as e:
            logger.error("Unexpected error sending PagerDuty alert: %s", e)
        
        return False


class WebhookNotifier(BaseNotifier):
    """Send alerts to a generic webhook endpoint."""
    
    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        
    def send(self, payload: AlertPayload) -> bool:
        if not self.webhook_url:
            logger.error("Webhook URL not configured")
            return False
        
        try:
            data = json.dumps(payload.to_dict()).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urlopen(req, timeout=10) as response:
                if response.status in (200, 201, 202, 204):
                    logger.info("Webhook alert sent successfully")
                    return True
                    
        except URLError as e:
            logger.error("Failed to send webhook alert: %s", e)
        except Exception as e:
            logger.error("Unexpected error sending webhook alert: %s", e)
        
        return False


class LogNotifier(BaseNotifier):
    """Log alerts (useful for development/testing)."""
    
    def send(self, payload: AlertPayload) -> bool:
        logger.warning(
            "ALERT [%s] %s - %s (model=%s)",
            payload.severity.upper(),
            payload.title,
            payload.message,
            payload.model_version,
        )
        return True


class AlertNotifier:
    """
    Multi-channel alert notification manager.
    
    Sends alerts to configured channels based on severity and configuration.
    """
    
    def __init__(self) -> None:
        self._notifiers: dict[AlertChannel, BaseNotifier] = {}
        self._severity_channels: dict[DriftSeverity, list[AlertChannel]] = {
            DriftSeverity.NONE: [],
            DriftSeverity.LOW: [AlertChannel.LOG],
            DriftSeverity.MODERATE: [AlertChannel.LOG, AlertChannel.SLACK],
            DriftSeverity.HIGH: [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            DriftSeverity.CRITICAL: [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
        }
        
    def register_channel(self, channel: AlertChannel, notifier: BaseNotifier) -> None:
        """Register a notifier for a channel."""
        self._notifiers[channel] = notifier
        logger.info("Registered alert channel: %s", channel.value)
        
    def configure_severity_routing(
        self,
        severity: DriftSeverity,
        channels: list[AlertChannel],
    ) -> None:
        """Configure which channels receive alerts for each severity level."""
        self._severity_channels[severity] = channels
        
    def send_drift_alert(self, drift_result: DriftResult) -> dict[AlertChannel, bool]:
        """
        Send drift alert to appropriate channels.
        
        Returns dict mapping channels to success status.
        """
        if not drift_result.alert_triggered:
            return {}
        
        drifted_features = [
            f.feature_name for f in drift_result.feature_results
            if f.severity != DriftSeverity.NONE
        ]
        
        payload = AlertPayload(
            title=f"Model Drift Detected - {drift_result.overall_severity.value.upper()}",
            message=f"Drift detected in {len(drifted_features)} feature(s): {', '.join(drifted_features[:5])}",
            severity=drift_result.overall_severity.value,
            timestamp=drift_result.timestamp,
            model_version=drift_result.model_version,
            details={
                "drifted_features": len(drifted_features),
                "reference_samples": drift_result.reference_size,
                "current_samples": drift_result.current_size,
                "max_psi": max(f.psi for f in drift_result.feature_results) if drift_result.feature_results else 0,
            },
        )
        
        channels = self._severity_channels.get(drift_result.overall_severity, [])
        results: dict[AlertChannel, bool] = {}
        
        for channel in channels:
            notifier = self._notifiers.get(channel)
            if notifier:
                results[channel] = notifier.send(payload)
            else:
                logger.warning("No notifier registered for channel: %s", channel.value)
                results[channel] = False
        
        return results
    
    def send_custom_alert(
        self,
        title: str,
        message: str,
        severity: DriftSeverity,
        model_version: str,
        details: dict[str, Any] | None = None,
    ) -> dict[AlertChannel, bool]:
        """Send a custom alert to channels configured for the severity level."""
        payload = AlertPayload(
            title=title,
            message=message,
            severity=severity.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=model_version,
            details=details or {},
        )
        
        channels = self._severity_channels.get(severity, [])
        results: dict[AlertChannel, bool] = {}
        
        for channel in channels:
            notifier = self._notifiers.get(channel)
            if notifier:
                results[channel] = notifier.send(payload)
            else:
                results[channel] = False
        
        return results


def create_default_notifier() -> AlertNotifier:
    """Create an AlertNotifier with default channel configuration."""
    notifier = AlertNotifier()
    
    notifier.register_channel(AlertChannel.LOG, LogNotifier())
    
    if os.getenv("SLACK_WEBHOOK_URL"):
        notifier.register_channel(AlertChannel.SLACK, SlackNotifier())
    
    if os.getenv("PAGERDUTY_ROUTING_KEY"):
        notifier.register_channel(AlertChannel.PAGERDUTY, PagerDutyNotifier())
    
    if os.getenv("ALERT_WEBHOOK_URL"):
        notifier.register_channel(AlertChannel.WEBHOOK, WebhookNotifier())
    
    return notifier
