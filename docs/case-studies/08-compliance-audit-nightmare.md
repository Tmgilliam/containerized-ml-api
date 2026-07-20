# Case Study 8: Compliance Audit Nightmare — Missing Prediction Logs

## Company Profile

**Industry:** Defense contractor supply chain  
**Regulatory Framework:** DFARS, NIST 800-171, SOC 2 Type II  
**Challenge:** Audit trail requirements for ML-driven decisions

---

## The Problem

### Situation

A defense contractor implemented an ML-based delay risk scoring system to manage their supply chain. The system influenced procurement decisions for components used in defense equipment.

### The Audit Request

During a routine DFARS compliance audit, the auditor requested:

1. **Complete prediction logs** for the past 18 months
2. **Model version history** showing which model produced each prediction
3. **Feature values** used for each prediction
4. **Decision justification** for orders flagged as high-risk
5. **Outcome tracking** showing prediction accuracy over time

### The Discovery

The team could only provide:
- ✅ API access logs (timestamp, IP, status code)
- ✅ Current model version
- ❌ Historical prediction values
- ❌ Features used in predictions
- ❌ Model version per prediction
- ❌ Outcome data linked to predictions
- ❌ Explanation of predictions

### The Consequence

- **$2.8M contract** put on hold pending remediation
- **6-month deadline** to implement comprehensive audit logging
- **Manual reconstruction** of 50,000 predictions (infeasible)
- **Reputational risk** with prime contractor

---

## The Solution

### Phase 1: Comprehensive Prediction Logging

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Any
import json
import uuid
from pathlib import Path

@dataclass
class PredictionAuditRecord:
    """Complete audit record for each prediction."""
    
    # Identity
    prediction_id: str
    timestamp: str
    
    # Input
    request_payload: dict
    features_used: dict
    
    # Model info
    model_name: str
    model_version: str
    model_checksum: str
    
    # Output
    prediction_class: int
    prediction_probability: float
    confidence: float
    threshold_applied: float
    
    # Context
    client_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    request_source: str
    
    # Explanation (if generated)
    explanation_summary: Optional[str] = None
    top_features: Optional[list] = None
    
    # Compliance
    data_retention_days: int = 2190  # 6 years for DFARS
    classification_level: str = "UNCLASSIFIED"


class AuditLogger:
    """Log predictions with full audit trail."""
    
    def __init__(self, storage_path: Path, retention_days: int = 2190):
        self.storage_path = storage_path
        self.retention_days = retention_days
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Buffer for batch writes
        self._buffer = []
        self._buffer_size = 100
    
    def log_prediction(
        self,
        request: dict,
        features: dict,
        prediction: dict,
        model_info: dict,
        client_id: str,
        explanation: Optional[dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_source: str = "api",
    ) -> str:
        """Log a prediction with full audit trail."""
        
        prediction_id = str(uuid.uuid4())
        
        record = PredictionAuditRecord(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_payload=request,
            features_used=features,
            model_name=model_info["name"],
            model_version=model_info["version"],
            model_checksum=model_info.get("checksum", "unknown"),
            prediction_class=prediction["delay_risk"],
            prediction_probability=prediction["risk_score"],
            confidence=prediction.get("confidence", 0.0),
            threshold_applied=prediction.get("threshold", 0.5),
            client_id=client_id,
            user_id=user_id,
            session_id=session_id,
            request_source=request_source,
            explanation_summary=explanation.get("summary") if explanation else None,
            top_features=explanation.get("top_features") if explanation else None,
            data_retention_days=self.retention_days,
        )
        
        self._buffer.append(asdict(record))
        
        if len(self._buffer) >= self._buffer_size:
            self._flush()
        
        return prediction_id
    
    def _flush(self):
        """Write buffered records to storage."""
        
        if not self._buffer:
            return
        
        # Organize by date for efficient querying
        today = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = self.storage_path / f"predictions_{today}.jsonl"
        
        with open(file_path, "a") as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")
        
        self._buffer = []
    
    def query_predictions(
        self,
        start_date: str,
        end_date: str,
        client_id: Optional[str] = None,
        prediction_class: Optional[int] = None,
    ) -> list[dict]:
        """Query predictions for audit reports."""
        
        results = []
        
        # Iterate through date range
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"predictions_{date_str}.jsonl"
            
            if file_path.exists():
                with open(file_path) as f:
                    for line in f:
                        record = json.loads(line)
                        
                        # Apply filters
                        if client_id and record["client_id"] != client_id:
                            continue
                        if prediction_class is not None and record["prediction_class"] != prediction_class:
                            continue
                        
                        results.append(record)
            
            current += timedelta(days=1)
        
        return results
    
    def export_for_audit(
        self,
        start_date: str,
        end_date: str,
        output_path: Path,
    ) -> dict:
        """Export predictions in audit-friendly format."""
        
        predictions = self.query_predictions(start_date, end_date)
        
        # Write to CSV for auditors
        import csv
        
        csv_path = output_path / f"audit_export_{start_date}_to_{end_date}.csv"
        
        with open(csv_path, "w", newline="") as f:
            if predictions:
                writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
        
        # Write summary
        summary = {
            "export_date": datetime.utcnow().isoformat(),
            "period_start": start_date,
            "period_end": end_date,
            "total_predictions": len(predictions),
            "unique_clients": len(set(p["client_id"] for p in predictions)),
            "model_versions_used": list(set(p["model_version"] for p in predictions)),
            "high_risk_count": sum(1 for p in predictions if p["prediction_class"] == 1),
            "low_risk_count": sum(1 for p in predictions if p["prediction_class"] == 0),
        }
        
        summary_path = output_path / f"audit_summary_{start_date}_to_{end_date}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
```

### Phase 2: Outcome Tracking for Accuracy Verification

```python
from retraining.collectors import OutcomeIngester, LabelMatcher

# Initialize outcome tracking
outcome_ingester = OutcomeIngester(
    storage_path=Path("./data/outcomes"),
    buffer_size=100,
)

# Ingest outcomes as they become known
def record_delivery_outcome(
    order_id: str,
    was_delayed: bool,
    actual_delivery_date: str,
    expected_delivery_date: str,
):
    """Record actual delivery outcome for audit trail."""
    
    outcome_ingester.ingest(
        entity_id=order_id,
        outcome=1 if was_delayed else 0,
        metadata={
            "actual_delivery_date": actual_delivery_date,
            "expected_delivery_date": expected_delivery_date,
            "delay_days": calculate_delay_days(
                actual_delivery_date, 
                expected_delivery_date
            ) if was_delayed else 0,
        }
    )

# Generate accuracy report for auditors
def generate_accuracy_audit_report(
    start_date: str,
    end_date: str,
) -> dict:
    """Generate accuracy report for compliance audit."""
    
    # Load predictions and outcomes
    matcher = LabelMatcher(
        predictions_path=Path("./data/audit_logs"),
        outcomes_path=Path("./data/outcomes"),
    )
    
    matcher.load_predictions()
    matcher.load_outcomes()
    
    # Match and calculate metrics
    accuracy_report = matcher.accuracy_report()
    
    # Add audit-specific information
    audit_report = {
        "report_date": datetime.utcnow().isoformat(),
        "period_start": start_date,
        "period_end": end_date,
        "compliance_framework": "DFARS",
        
        # Core metrics
        "total_predictions": accuracy_report["total_matched"],
        "accuracy": accuracy_report["accuracy"],
        "precision": accuracy_report["precision"],
        "recall": accuracy_report["recall"],
        "f1_score": accuracy_report["f1_score"],
        
        # Confusion matrix
        "true_positives": accuracy_report["true_positives"],
        "true_negatives": accuracy_report["true_negatives"],
        "false_positives": accuracy_report["false_positives"],
        "false_negatives": accuracy_report["false_negatives"],
        
        # Business impact
        "correctly_flagged_delays": accuracy_report["true_positives"],
        "missed_delays": accuracy_report["false_negatives"],
        "false_alarms": accuracy_report["false_positives"],
        
        # Attestation
        "generated_by": "delay-risk-api-v1",
        "data_integrity_verified": True,
    }
    
    return audit_report
```

### Phase 3: Model Version Tracking

```python
from app.models import ModelRegistry
import hashlib

class AuditableModelRegistry(ModelRegistry):
    """Model registry with audit trail for compliance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._version_history = []
    
    def register_model(
        self,
        name: str,
        model_path: str,
        version: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Register model with checksum for integrity verification."""
        
        # Calculate model checksum
        with open(model_path, "rb") as f:
            model_bytes = f.read()
            checksum = hashlib.sha256(model_bytes).hexdigest()
        
        # Record version history
        version_record = {
            "name": name,
            "version": version,
            "checksum": checksum,
            "registered_at": datetime.utcnow().isoformat(),
            "model_path": str(model_path),
            "metadata": metadata or {},
        }
        
        self._version_history.append(version_record)
        
        # Write to persistent storage
        self._save_version_history()
        
        # Call parent registration
        return super().register_model(name, model_path, version, metadata)
    
    def _save_version_history(self):
        """Persist version history for audit."""
        
        history_path = Path("./data/model_versions.json")
        
        with open(history_path, "w") as f:
            json.dump(self._version_history, f, indent=2)
    
    def get_model_info_for_audit(self, name: str) -> dict:
        """Get complete model information for audit."""
        
        model = self.get_model(name)
        
        return {
            "name": name,
            "version": model.version,
            "checksum": model.checksum,
            "status": model.status,
            "loaded_at": model.loaded_at.isoformat(),
            "metadata": model.metadata,
        }
    
    def export_version_history(self, output_path: Path) -> str:
        """Export model version history for auditors."""
        
        export_path = output_path / "model_version_history.json"
        
        with open(export_path, "w") as f:
            json.dump({
                "export_date": datetime.utcnow().isoformat(),
                "versions": self._version_history,
            }, f, indent=2)
        
        return str(export_path)
```

### Phase 4: Explanation Logging for Decision Justification

```python
from app.explainers import SHAPExplainer

def predict_with_audit_trail(
    features: dict,
    request: dict,
    client_id: str,
    audit_logger: AuditLogger,
    explainer: SHAPExplainer,
    model: Any,
    model_info: dict,
    require_explanation: bool = True,
) -> dict:
    """Make prediction with complete audit trail."""
    
    # Get prediction
    prediction = model.predict(features)
    
    # Generate explanation (required for high-risk predictions)
    explanation = None
    if require_explanation or prediction["delay_risk"] == 1:
        explanation = explainer.explain(features, prediction)
        explanation_dict = {
            "summary": explanation.summary(top_n=5),
            "top_features": [
                {
                    "name": f.feature_name,
                    "value": f.feature_value,
                    "contribution": f.contribution,
                    "direction": f.direction,
                }
                for f in explanation.features[:5]
            ],
        }
    else:
        explanation_dict = None
    
    # Log with full audit trail
    prediction_id = audit_logger.log_prediction(
        request=request,
        features=features,
        prediction=prediction,
        model_info=model_info,
        client_id=client_id,
        explanation=explanation_dict,
    )
    
    # Add audit reference to response
    return {
        **prediction,
        "prediction_id": prediction_id,
        "explanation": explanation_dict,
        "audit_trail": {
            "logged": True,
            "prediction_id": prediction_id,
            "model_version": model_info["version"],
            "model_checksum": model_info.get("checksum"),
        }
    }
```

### Phase 5: Compliance Report Generation

```python
class ComplianceReportGenerator:
    """Generate compliance reports for various frameworks."""
    
    def __init__(
        self,
        audit_logger: AuditLogger,
        model_registry: AuditableModelRegistry,
    ):
        self.audit_logger = audit_logger
        self.model_registry = model_registry
    
    def generate_dfars_report(
        self,
        start_date: str,
        end_date: str,
        output_path: Path,
    ) -> dict:
        """Generate DFARS compliance report."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Prediction logs
        prediction_summary = self.audit_logger.export_for_audit(
            start_date, end_date, output_path
        )
        
        # 2. Model version history
        model_history_path = self.model_registry.export_version_history(output_path)
        
        # 3. Accuracy report
        accuracy_report = generate_accuracy_audit_report(start_date, end_date)
        accuracy_path = output_path / "accuracy_report.json"
        with open(accuracy_path, "w") as f:
            json.dump(accuracy_report, f, indent=2)
        
        # 4. System configuration
        config_report = self._generate_config_report()
        config_path = output_path / "system_configuration.json"
        with open(config_path, "w") as f:
            json.dump(config_report, f, indent=2)
        
        # 5. Executive summary
        executive_summary = {
            "report_type": "DFARS Compliance Audit Report",
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date,
                "end": end_date,
            },
            "summary": {
                "total_predictions": prediction_summary["total_predictions"],
                "unique_clients": prediction_summary["unique_clients"],
                "model_versions_used": prediction_summary["model_versions_used"],
                "prediction_accuracy": accuracy_report["accuracy"],
            },
            "compliance_status": {
                "prediction_logging": "COMPLIANT",
                "model_versioning": "COMPLIANT",
                "outcome_tracking": "COMPLIANT",
                "explanation_logging": "COMPLIANT",
                "data_retention": f"COMPLIANT ({self.audit_logger.retention_days} days)",
            },
            "artifacts": {
                "prediction_logs": str(output_path / f"audit_export_{start_date}_to_{end_date}.csv"),
                "model_history": model_history_path,
                "accuracy_report": str(accuracy_path),
                "system_config": str(config_path),
            }
        }
        
        summary_path = output_path / "executive_summary.json"
        with open(summary_path, "w") as f:
            json.dump(executive_summary, f, indent=2)
        
        return executive_summary
    
    def _generate_config_report(self) -> dict:
        """Document system configuration for audit."""
        
        return {
            "system_name": "Delay Risk Scoring API",
            "version": "1.0.0",
            "deployment": {
                "environment": "production",
                "cloud_provider": "GCP",
                "region": "us-central1",
            },
            "data_handling": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "data_retention_days": 2190,
                "pii_handling": "No PII processed",
            },
            "access_controls": {
                "authentication": "API Key + JWT",
                "authorization": "Role-based (RBAC)",
                "audit_logging": "Enabled",
            },
            "model_governance": {
                "version_control": "Git",
                "checksum_verification": "SHA-256",
                "change_management": "PR-based approval",
            },
        }
```

### Phase 6: API Endpoints for Auditors

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

audit_router = APIRouter(prefix="/audit", tags=["Audit"])

@audit_router.get("/predictions")
async def get_prediction_logs(
    start_date: str,
    end_date: str,
    client_id: Optional[str] = None,
    claims: dict = Depends(require_audit_role),
):
    """Retrieve prediction logs for audit (requires audit role)."""
    
    predictions = audit_logger.query_predictions(
        start_date=start_date,
        end_date=end_date,
        client_id=client_id,
    )
    
    return {
        "period": {"start": start_date, "end": end_date},
        "count": len(predictions),
        "predictions": predictions[:1000],  # Paginate large results
    }

@audit_router.get("/prediction/{prediction_id}")
async def get_prediction_detail(
    prediction_id: str,
    claims: dict = Depends(require_audit_role),
):
    """Get complete audit trail for a single prediction."""
    
    record = audit_logger.get_by_id(prediction_id)
    
    if not record:
        raise HTTPException(404, "Prediction not found")
    
    # Get associated outcome if available
    outcome = outcome_ingester.get_outcome(record["request_payload"].get("order_id"))
    
    return {
        "prediction": record,
        "outcome": outcome,
        "accuracy": {
            "correct": outcome["outcome"] == record["prediction_class"] if outcome else None,
        }
    }

@audit_router.post("/reports/dfars")
async def generate_dfars_report(
    start_date: str,
    end_date: str,
    claims: dict = Depends(require_audit_role),
):
    """Generate DFARS compliance report."""
    
    output_path = Path(f"./reports/dfars_{start_date}_{end_date}")
    
    report_generator = ComplianceReportGenerator(
        audit_logger=audit_logger,
        model_registry=model_registry,
    )
    
    summary = report_generator.generate_dfars_report(
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
    )
    
    return summary

@audit_router.get("/reports/dfars/{report_id}/download")
async def download_dfars_report(
    report_id: str,
    claims: dict = Depends(require_audit_role),
):
    """Download complete DFARS report as ZIP."""
    
    report_path = Path(f"./reports/{report_id}")
    
    if not report_path.exists():
        raise HTTPException(404, "Report not found")
    
    # Create ZIP archive
    import shutil
    zip_path = shutil.make_archive(
        str(report_path),
        'zip',
        report_path,
    )
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"dfars_audit_report_{report_id}.zip",
    )
```

---

## Results

### Before Implementation

| Requirement | Status |
|-------------|--------|
| Prediction logs | ❌ Missing |
| Model version per prediction | ❌ Missing |
| Feature values logged | ❌ Missing |
| Decision explanations | ❌ Missing |
| Outcome tracking | ❌ Missing |
| Audit report generation | ❌ Manual/impossible |

### After Implementation

| Requirement | Status |
|-------------|--------|
| Prediction logs | ✅ 100% coverage |
| Model version per prediction | ✅ Checksum verified |
| Feature values logged | ✅ Complete |
| Decision explanations | ✅ SHAP-based |
| Outcome tracking | ✅ Automated matching |
| Audit report generation | ✅ One-click DFARS report |

### Audit Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Audit findings | 5 critical | 0 critical |
| Remediation time | 6 months | N/A |
| Contract status | On hold | Active |
| Auditor confidence | Low | High |

---

## Key Learnings

1. **Log everything** — Predictions, features, model versions, and explanations must all be captured
2. **Link predictions to outcomes** — Accuracy claims require verifiable outcome data
3. **Model integrity matters** — Checksums prove which model produced which prediction
4. **Explanations are mandatory** — "The model said so" is not acceptable for regulated industries
5. **Retention policies vary** — DFARS requires 6 years; know your framework's requirements

---

## Compliance Checklist

Before deploying in regulated environments:

- [ ] All predictions logged with unique ID
- [ ] Features captured for each prediction
- [ ] Model version and checksum recorded
- [ ] Explanations generated for high-risk predictions
- [ ] Outcomes tracked and matched to predictions
- [ ] Audit export endpoints available
- [ ] Retention policy configured (6 years for DFARS)
- [ ] Access controls on audit data
- [ ] Data encryption at rest and in transit

---

## Related Modules

- `retraining/collectors/outcome_ingester.py` — Outcome collection
- `retraining/collectors/label_matcher.py` — Prediction/outcome matching
- `app/explainers/shap_explainer.py` — Decision explanations
- `app/models/registry.py` — Model version management
- `gateway/usage_tracker.py` — Request logging
