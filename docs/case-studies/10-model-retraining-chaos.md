# Case Study 10: Model Retraining Without Guardrails Causes Production Incident

## Company Profile

**Industry:** Electronics contract manufacturer  
**Scale:** 25,000 orders/day across 150 suppliers  
**Challenge:** Automated retraining without proper validation

---

## The Problem

### Situation

An electronics manufacturer implemented automated model retraining to keep their delay risk model fresh. The pipeline ran weekly, incorporating the latest outcome data.

### The Automation

```python
# The original (problematic) automation
def weekly_retrain():
    """PROBLEMATIC: No guardrails."""
    
    # 1. Get latest training data
    training_data = fetch_training_data(days=90)
    
    # 2. Train new model
    new_model = train_model(training_data)
    
    # 3. Deploy directly to production (!)
    deploy_model(new_model)
    
    print("Retraining complete!")
```

### The Incident

**Week 12:** The automated pipeline ran as scheduled. The new model was deployed at 3:00 AM.

**6:00 AM:** Procurement teams reported that 95% of orders were flagged as "high risk."

**Investigation revealed:**

1. A data pipeline bug corrupted 2 weeks of outcome labels
2. 73% of "on-time" orders were mislabeled as "delayed"
3. The model learned that "delayed" was the norm
4. The new model predicted "delay" for almost everything

### The Impact

| Metric | Normal | During Incident |
|--------|--------|-----------------|
| High-risk predictions | 18% | 95% |
| Procurement workload | Normal | 5x (investigating false alarms) |
| Supplier calls | 200/day | 1,100/day |
| Trust in system | High | Destroyed |

### Recovery

- **4 hours** to identify the issue
- **2 hours** to roll back to previous model
- **2 weeks** to rebuild trust with procurement team
- **3 months** to fix data pipeline and implement guardrails

---

## The Solution

### Phase 1: Outcome Validation Before Training

```python
from retraining.collectors import LabelMatcher

def validate_training_data(training_data: pd.DataFrame) -> dict:
    """Validate training data before using for retraining."""
    
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
    }
    
    # Check 1: Label distribution
    delay_rate = training_data["was_delayed"].mean()
    
    if delay_rate > 0.5:
        validation_results["errors"].append({
            "check": "label_distribution",
            "message": f"Abnormal delay rate: {delay_rate:.1%} (expected 15-25%)",
            "value": delay_rate,
        })
        validation_results["is_valid"] = False
    
    elif delay_rate > 0.3 or delay_rate < 0.1:
        validation_results["warnings"].append({
            "check": "label_distribution",
            "message": f"Unusual delay rate: {delay_rate:.1%}",
            "value": delay_rate,
        })
    
    # Check 2: Missing values
    missing_rate = training_data.isnull().sum() / len(training_data)
    high_missing = missing_rate[missing_rate > 0.1]
    
    if len(high_missing) > 0:
        validation_results["warnings"].append({
            "check": "missing_values",
            "message": f"High missing rate in columns: {list(high_missing.index)}",
            "values": high_missing.to_dict(),
        })
    
    # Check 3: Sample size
    if len(training_data) < 10000:
        validation_results["errors"].append({
            "check": "sample_size",
            "message": f"Insufficient training data: {len(training_data)} (min 10,000)",
            "value": len(training_data),
        })
        validation_results["is_valid"] = False
    
    # Check 4: Feature drift from reference
    reference_stats = load_reference_feature_stats()
    for feature in reference_stats:
        if feature in training_data.columns:
            current_mean = training_data[feature].mean()
            ref_mean = reference_stats[feature]["mean"]
            ref_std = reference_stats[feature]["std"]
            
            z_score = abs(current_mean - ref_mean) / ref_std if ref_std > 0 else 0
            
            if z_score > 3:
                validation_results["errors"].append({
                    "check": "feature_drift",
                    "feature": feature,
                    "message": f"Extreme drift in {feature}: z-score = {z_score:.2f}",
                })
                validation_results["is_valid"] = False
    
    return validation_results
```

### Phase 2: Model Validation Before Deployment

```python
from retraining.pipelines import TrainingPipeline

def validate_model_quality(
    new_model,
    validation_data: pd.DataFrame,
    production_model,
) -> dict:
    """Validate new model quality before deployment."""
    
    validation_results = {
        "is_valid": True,
        "metrics": {},
        "comparison": {},
        "warnings": [],
        "errors": [],
    }
    
    X_val = validation_data[FEATURE_COLUMNS]
    y_val = validation_data["was_delayed"]
    
    # Evaluate new model
    new_predictions = new_model.predict_proba(X_val)[:, 1]
    new_metrics = calculate_metrics(y_val, new_predictions)
    
    # Evaluate production model
    prod_predictions = production_model.predict_proba(X_val)[:, 1]
    prod_metrics = calculate_metrics(y_val, prod_predictions)
    
    validation_results["metrics"] = {
        "new_model": new_metrics,
        "production_model": prod_metrics,
    }
    
    # Check 1: Minimum performance thresholds
    min_thresholds = {
        "auc": 0.75,
        "precision": 0.60,
        "recall": 0.70,
    }
    
    for metric, threshold in min_thresholds.items():
        if new_metrics[metric] < threshold:
            validation_results["errors"].append({
                "check": "minimum_threshold",
                "metric": metric,
                "message": f"{metric} below minimum: {new_metrics[metric]:.3f} < {threshold}",
            })
            validation_results["is_valid"] = False
    
    # Check 2: No significant regression from production
    regression_thresholds = {
        "auc": 0.02,      # Max 2% drop allowed
        "precision": 0.05,
        "recall": 0.05,
    }
    
    for metric, max_drop in regression_thresholds.items():
        drop = prod_metrics[metric] - new_metrics[metric]
        
        if drop > max_drop:
            validation_results["errors"].append({
                "check": "regression",
                "metric": metric,
                "message": f"{metric} regression: {drop:.3f} > {max_drop} allowed",
                "production_value": prod_metrics[metric],
                "new_value": new_metrics[metric],
            })
            validation_results["is_valid"] = False
        
        elif drop > max_drop / 2:
            validation_results["warnings"].append({
                "check": "regression",
                "metric": metric,
                "message": f"{metric} slight regression: {drop:.3f}",
            })
    
    # Check 3: Prediction distribution sanity
    new_positive_rate = (new_predictions >= 0.5).mean()
    prod_positive_rate = (prod_predictions >= 0.5).mean()
    
    rate_change = abs(new_positive_rate - prod_positive_rate)
    
    if rate_change > 0.2:
        validation_results["errors"].append({
            "check": "prediction_distribution",
            "message": f"Abnormal prediction rate change: {rate_change:.1%}",
            "production_rate": prod_positive_rate,
            "new_rate": new_positive_rate,
        })
        validation_results["is_valid"] = False
    
    validation_results["comparison"] = {
        "auc_change": new_metrics["auc"] - prod_metrics["auc"],
        "precision_change": new_metrics["precision"] - prod_metrics["precision"],
        "recall_change": new_metrics["recall"] - prod_metrics["recall"],
        "positive_rate_change": new_positive_rate - prod_positive_rate,
    }
    
    return validation_results
```

### Phase 3: Automated Retraining with Guardrails

```python
from retraining.triggers import RetrainTrigger
from retraining.pipelines import TrainingPipeline

class SafeRetrainingPipeline:
    """Retraining pipeline with comprehensive guardrails."""
    
    def __init__(
        self,
        model_registry,
        notification_callback,
        require_approval: bool = True,
    ):
        self.model_registry = model_registry
        self.notification_callback = notification_callback
        self.require_approval = require_approval
    
    def run(self) -> dict:
        """Execute safe retraining pipeline."""
        
        result = {
            "status": "pending",
            "stages": {},
            "final_action": None,
        }
        
        # Stage 1: Check if retraining is needed
        trigger = RetrainTrigger()
        trigger_result = trigger.check_all_triggers()
        
        result["stages"]["trigger_check"] = trigger_result
        
        if not trigger_result.should_retrain:
            result["status"] = "skipped"
            result["final_action"] = "No retraining needed"
            return result
        
        # Stage 2: Prepare training data
        training_data = self._prepare_training_data()
        
        # Stage 3: Validate training data
        data_validation = validate_training_data(training_data)
        result["stages"]["data_validation"] = data_validation
        
        if not data_validation["is_valid"]:
            result["status"] = "blocked"
            result["final_action"] = "Training data failed validation"
            
            self.notification_callback({
                "type": "RETRAINING_BLOCKED",
                "reason": "Data validation failed",
                "errors": data_validation["errors"],
            })
            
            return result
        
        # Stage 4: Train new model
        pipeline = TrainingPipeline(
            model_name="delay-risk-model",
            output_dir=Path("./models/candidates"),
        )
        
        X = training_data[FEATURE_COLUMNS]
        y = training_data["was_delayed"]
        
        training_result = pipeline.train(X=X, y=y, test_size=0.2, cv_folds=5)
        result["stages"]["training"] = {
            "metrics": training_result.metrics,
            "cv_scores": training_result.cv_scores,
        }
        
        # Stage 5: Validate model quality
        validation_data = self._prepare_validation_data()
        production_model = self.model_registry.get_model("delay-risk-model")
        
        model_validation = validate_model_quality(
            new_model=training_result.model,
            validation_data=validation_data,
            production_model=production_model.model,
        )
        result["stages"]["model_validation"] = model_validation
        
        if not model_validation["is_valid"]:
            result["status"] = "blocked"
            result["final_action"] = "Model failed validation"
            
            self.notification_callback({
                "type": "RETRAINING_BLOCKED",
                "reason": "Model validation failed",
                "errors": model_validation["errors"],
                "metrics": model_validation["metrics"],
            })
            
            return result
        
        # Stage 6: Shadow mode testing (optional)
        shadow_result = self._run_shadow_test(training_result.model)
        result["stages"]["shadow_test"] = shadow_result
        
        if shadow_result and not shadow_result["passed"]:
            result["status"] = "blocked"
            result["final_action"] = "Shadow test failed"
            return result
        
        # Stage 7: Request approval or auto-deploy
        if self.require_approval:
            result["status"] = "pending_approval"
            result["final_action"] = "Awaiting human approval"
            
            self.notification_callback({
                "type": "RETRAINING_APPROVAL_REQUESTED",
                "metrics_comparison": model_validation["comparison"],
                "model_path": str(training_result.model_path),
            })
            
            # Save candidate model for later deployment
            self._save_candidate_model(training_result)
            
        else:
            # Auto-deploy with all checks passed
            self._deploy_model(training_result)
            result["status"] = "deployed"
            result["final_action"] = "Model deployed to production"
            
            self.notification_callback({
                "type": "RETRAINING_COMPLETE",
                "new_model_version": training_result.version,
                "metrics": model_validation["metrics"]["new_model"],
            })
        
        return result
    
    def _run_shadow_test(self, new_model, duration_hours: int = 4) -> dict:
        """Run shadow test before deployment."""
        
        # Configure shadow mode
        shadow_runner = ShadowRunner(
            production_model=self.model_registry.get_model("delay-risk-model").predict,
            shadow_model=new_model.predict_proba,
            config=ShadowConfig(
                enabled=True,
                sample_rate=1.0,
                async_execution=True,
            ),
        )
        
        # Run for specified duration (simplified - in practice use scheduled job)
        comparator = ShadowComparator(
            agreement_threshold=0.90,
            probability_diff_threshold=0.15,
        )
        
        # Simulate shadow test results
        is_ready, issues = comparator.is_shadow_ready_for_promotion()
        
        return {
            "passed": is_ready,
            "issues": issues,
            "agreement_rate": comparator.analyze().prediction_agreement_rate,
        }
    
    def approve_and_deploy(self, candidate_id: str) -> dict:
        """Deploy approved candidate model."""
        
        candidate = self._load_candidate_model(candidate_id)
        
        if not candidate:
            return {"error": "Candidate not found"}
        
        self._deploy_model(candidate)
        
        return {
            "status": "deployed",
            "model_version": candidate.version,
        }
```

### Phase 4: Trigger-Based Retraining

```python
from retraining.triggers import RetrainTrigger, TriggerResult

# Configure triggers
trigger = RetrainTrigger()

# Trigger 1: Drift-based
trigger.add_drift_trigger(
    name="feature_drift",
    drift_detector=drift_detector,
    threshold_severity=DriftSeverity.HIGH,
)

# Trigger 2: Accuracy-based
trigger.add_accuracy_trigger(
    name="accuracy_drop",
    accuracy_threshold=0.80,
    sample_size_threshold=1000,
)

# Trigger 3: Scheduled (weekly)
trigger.add_schedule_trigger(
    name="weekly_retrain",
    cron_expression="0 2 * * 0",  # Sunday 2 AM
)

# Trigger 4: Data volume-based
trigger.add_volume_trigger(
    name="new_data_volume",
    new_records_threshold=10000,
)

# Check all triggers
def check_retraining_needed():
    """Check if any trigger fires."""
    
    result = trigger.check_all_triggers()
    
    return {
        "should_retrain": result.should_retrain,
        "active_triggers": result.active_triggers,
        "trigger_details": result.details,
    }
```

### Phase 5: Rollback Capability

```python
class ModelRollbackManager:
    """Manage model rollbacks when issues are detected."""
    
    def __init__(self, model_registry):
        self.model_registry = model_registry
        self._deployment_history = []
    
    def record_deployment(
        self,
        model_name: str,
        model_version: str,
        previous_version: str,
    ):
        """Record deployment for rollback capability."""
        
        self._deployment_history.append({
            "model_name": model_name,
            "deployed_version": model_version,
            "previous_version": previous_version,
            "deployed_at": datetime.now().isoformat(),
            "rolled_back": False,
        })
    
    def rollback(self, model_name: str, reason: str) -> dict:
        """Rollback to previous model version."""
        
        # Find last deployment
        deployments = [
            d for d in self._deployment_history
            if d["model_name"] == model_name and not d["rolled_back"]
        ]
        
        if not deployments:
            return {"error": "No deployment to rollback"}
        
        last_deployment = deployments[-1]
        previous_version = last_deployment["previous_version"]
        
        # Perform rollback
        self.model_registry.set_active_version(model_name, previous_version)
        
        # Record rollback
        last_deployment["rolled_back"] = True
        last_deployment["rollback_reason"] = reason
        last_deployment["rolled_back_at"] = datetime.now().isoformat()
        
        return {
            "status": "rolled_back",
            "from_version": last_deployment["deployed_version"],
            "to_version": previous_version,
            "reason": reason,
        }
    
    def auto_rollback_check(self, model_name: str) -> dict:
        """Automatically check if rollback is needed."""
        
        # Check prediction distribution
        recent_predictions = fetch_recent_predictions(model_name, hours=1)
        
        if not recent_predictions:
            return {"action": "none", "reason": "No recent predictions"}
        
        positive_rate = sum(
            1 for p in recent_predictions if p["prediction_class"] == 1
        ) / len(recent_predictions)
        
        # Alert if prediction rate is abnormal
        if positive_rate > 0.5:  # More than 50% high risk
            return {
                "action": "rollback_recommended",
                "reason": f"Abnormal positive rate: {positive_rate:.1%}",
                "threshold": "50%",
            }
        
        return {"action": "none", "reason": "Predictions within normal range"}


# Automated rollback integration
@app.on_event("startup")
async def start_rollback_monitor():
    """Start background rollback monitoring."""
    
    async def monitor_loop():
        while True:
            result = rollback_manager.auto_rollback_check("delay-risk-model")
            
            if result["action"] == "rollback_recommended":
                # Send alert for human decision
                send_alert({
                    "type": "ROLLBACK_RECOMMENDED",
                    "model": "delay-risk-model",
                    "reason": result["reason"],
                })
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    asyncio.create_task(monitor_loop())
```

### Phase 6: Retraining Dashboard

```python
@app.get("/retraining/status")
def get_retraining_status():
    """Get current retraining pipeline status."""
    
    return {
        "last_retrain": get_last_retrain_info(),
        "next_scheduled": get_next_scheduled_retrain(),
        "trigger_status": trigger.check_all_triggers().to_dict(),
        "candidate_models": list_candidate_models(),
        "deployment_history": get_deployment_history(limit=10),
    }

@app.post("/retraining/trigger")
def trigger_retraining(
    force: bool = False,
    claims: dict = Depends(require_admin_role),
):
    """Manually trigger retraining."""
    
    if not force:
        # Check if conditions are met
        trigger_result = trigger.check_all_triggers()
        
        if not trigger_result.should_retrain:
            return {
                "status": "skipped",
                "reason": "No triggers active. Use force=true to override.",
            }
    
    # Start retraining pipeline
    pipeline = SafeRetrainingPipeline(
        model_registry=model_registry,
        notification_callback=send_slack_notification,
        require_approval=True,
    )
    
    result = pipeline.run()
    
    return result

@app.post("/retraining/approve/{candidate_id}")
def approve_candidate(
    candidate_id: str,
    claims: dict = Depends(require_admin_role),
):
    """Approve and deploy candidate model."""
    
    pipeline = SafeRetrainingPipeline(
        model_registry=model_registry,
        notification_callback=send_slack_notification,
    )
    
    return pipeline.approve_and_deploy(candidate_id)

@app.post("/retraining/rollback")
def rollback_model(
    reason: str,
    claims: dict = Depends(require_admin_role),
):
    """Rollback to previous model version."""
    
    return rollback_manager.rollback("delay-risk-model", reason)
```

---

## Results

### Before Implementation

| Stage | Validation |
|-------|------------|
| Training data | ❌ None |
| Model quality | ❌ None |
| Regression check | ❌ None |
| Shadow testing | ❌ None |
| Human approval | ❌ None |
| Rollback capability | ❌ None |

### After Implementation

| Stage | Validation |
|-------|------------|
| Training data | ✅ Label distribution, missing values, feature drift |
| Model quality | ✅ Minimum thresholds, cross-validation |
| Regression check | ✅ Compare to production model |
| Shadow testing | ✅ 4-hour shadow mode |
| Human approval | ✅ Required for deployment |
| Rollback capability | ✅ One-click rollback |

### Incident Prevention

| Scenario | Before | After |
|----------|--------|-------|
| Corrupted labels | Deployed to production | Blocked by data validation |
| Degraded model | Deployed | Blocked by quality check |
| Abnormal predictions | Discovered by users | Auto-detected, alert sent |
| Recovery time | 6 hours | 2 minutes (rollback) |

---

## Key Learnings

1. **Validate training data** — Check label distributions, missing values, and feature drift
2. **Compare to production** — New model must not regress on key metrics
3. **Shadow test first** — Run on real traffic before promotion
4. **Require human approval** — Automated training ≠ automated deployment
5. **Enable rollback** — One-click rollback saves hours during incidents
6. **Monitor after deployment** — Auto-detect abnormal prediction patterns

---

## Retraining Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Safe Retraining Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Trigger    │──No──▶ Skip retraining                        │
│  │    Check     │                                               │
│  └──────┬───────┘                                               │
│         │ Yes                                                    │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │    Data      │──Fail──▶ Block + Alert                        │
│  │  Validation  │                                               │
│  └──────┬───────┘                                               │
│         │ Pass                                                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │    Train     │                                               │
│  │    Model     │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   Quality    │──Fail──▶ Block + Alert                        │
│  │  Validation  │                                               │
│  └──────┬───────┘                                               │
│         │ Pass                                                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   Shadow     │──Fail──▶ Block + Alert                        │
│  │    Test      │                                               │
│  └──────┬───────┘                                               │
│         │ Pass                                                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   Human      │──Reject──▶ Archive candidate                  │
│  │  Approval    │                                               │
│  └──────┬───────┘                                               │
│         │ Approve                                                │
│         ▼                                                        │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   Deploy     │────▶│   Monitor    │──Issue──▶ Rollback      │
│  │   Model      │     │   (Auto)     │                         │
│  └──────────────┘     └──────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Related Modules

- `retraining/triggers/retrain_trigger.py` — Trigger conditions
- `retraining/pipelines/train_pipeline.py` — Training execution
- `retraining/collectors/label_matcher.py` — Outcome matching
- `monitoring/drift/detector.py` — Drift detection
- `app/shadow/runner.py` — Shadow mode testing
- `app/models/registry.py` — Model version management
