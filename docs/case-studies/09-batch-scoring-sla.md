# Case Study 9: Batch Scoring Misses SLA Due to Scaling Issues

## Company Profile

**Industry:** Industrial chemicals distributor  
**Scale:** 45,000 orders scored daily for next-day planning  
**Constraint:** Batch scoring must complete by 6 AM for morning planning meetings

---

## The Problem

### Situation

An industrial chemicals distributor runs a nightly batch job to score all open orders for delay risk. The scores feed into the morning planning dashboard used by procurement and logistics teams.

### The SLA

- **Start time:** 2:00 AM (after ERP data sync)
- **Deadline:** 6:00 AM (before first planning meeting)
- **Window:** 4 hours to score 45,000 orders

### The Failure Pattern

| Week | Orders | Duration | SLA Met |
|------|--------|----------|---------|
| 1 | 42,000 | 2h 45m | ✅ |
| 2 | 44,000 | 3h 10m | ✅ |
| 3 | 48,000 | 4h 25m | ❌ |
| 4 | 51,000 | 5h 40m | ❌ |
| 5 | 53,000 | 7h 15m | ❌ |

### Root Cause Analysis

1. **Linear scaling** — Single-threaded processing scaled linearly with order count
2. **Holiday surge** — Q4 orders increased 30% from baseline
3. **No parallelism** — One order at a time, one API call at a time
4. **No batching** — API designed for single predictions, not bulk
5. **No retry logic** — Transient failures caused full restarts

---

## The Solution

### Phase 1: Parallel Batch Pipeline

```python
from batch import BatchPipeline, BatchConfig
from batch.io import LocalFileReader, LocalFileWriter
from concurrent.futures import ThreadPoolExecutor
import time

# Configure batch pipeline
config = BatchConfig(
    batch_size=100,  # Process 100 orders per API call
    max_workers=8,   # 8 parallel threads
    max_retries=3,
    retry_delay_seconds=5,
    timeout_seconds=30,
)

pipeline = BatchPipeline(config=config)

def run_nightly_scoring():
    """Run nightly batch scoring with parallelism."""
    
    start_time = time.time()
    
    # Configure I/O
    reader = LocalFileReader(
        file_path="./data/orders_to_score.csv",
        format="csv",
    )
    
    writer = LocalFileWriter(
        file_path="./data/scored_orders.csv",
        format="csv",
    )
    
    # Define prediction function for batches
    def score_batch(orders: list[dict]) -> list[dict]:
        """Score a batch of orders."""
        
        # Call batch endpoint
        response = requests.post(
            "http://localhost:8080/batch/predict",
            json={"orders": orders},
            timeout=30,
        )
        
        return response.json()["predictions"]
    
    # Run pipeline
    result = pipeline.run(
        reader=reader,
        writer=writer,
        process_fn=score_batch,
    )
    
    elapsed = time.time() - start_time
    
    return {
        "total_processed": result.total_processed,
        "successful": result.successful,
        "failed": result.failed,
        "elapsed_seconds": elapsed,
        "throughput_per_second": result.total_processed / elapsed,
    }
```

### Phase 2: Batch API Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchPredictRequest(BaseModel):
    orders: List[dict]

class BatchPredictResponse(BaseModel):
    predictions: List[dict]
    batch_id: str
    processed_count: int
    processing_time_ms: float

@app.post("/batch/predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Score multiple orders in a single request."""
    
    start = time.perf_counter()
    batch_id = str(uuid.uuid4())
    
    predictions = []
    
    # Process in parallel using thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(score_single_order, order)
            for order in request.orders
        ]
        
        for future in futures:
            try:
                predictions.append(future.result(timeout=10))
            except Exception as e:
                predictions.append({
                    "error": str(e),
                    "delay_risk": None,
                })
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return BatchPredictResponse(
        predictions=predictions,
        batch_id=batch_id,
        processed_count=len(predictions),
        processing_time_ms=elapsed_ms,
    )

def score_single_order(order: dict) -> dict:
    """Score a single order (called in parallel)."""
    
    features = extract_features(order)
    result = delay_risk_model.predict(features)
    
    return {
        "order_id": order.get("order_id"),
        **result,
    }
```

### Phase 3: Cloud-Native Readers and Writers

```python
from batch.io import GCSReader, GCSWriter, BigQueryReader, BigQueryWriter

# GCS-based batch processing
def run_gcs_batch():
    """Run batch with GCS input/output."""
    
    reader = GCSReader(
        bucket="delay-risk-data",
        blob_path="daily/orders_to_score.csv",
        format="csv",
    )
    
    writer = GCSWriter(
        bucket="delay-risk-data",
        blob_path=f"daily/scored_orders_{datetime.now().strftime('%Y%m%d')}.csv",
        format="csv",
    )
    
    return pipeline.run(reader=reader, writer=writer, process_fn=score_batch)

# BigQuery-based batch processing
def run_bigquery_batch():
    """Run batch with BigQuery input/output."""
    
    reader = BigQueryReader(
        project_id="my-project",
        query="""
            SELECT *
            FROM `my-project.supply_chain.orders`
            WHERE status = 'OPEN'
            AND created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
        """,
    )
    
    writer = BigQueryWriter(
        project_id="my-project",
        dataset_id="supply_chain",
        table_id="order_risk_scores",
        write_disposition="WRITE_TRUNCATE",  # Replace table
    )
    
    return pipeline.run(reader=reader, writer=writer, process_fn=score_batch)
```

### Phase 4: Checkpointing and Recovery

```python
from batch import CheckpointManager

class ResilientBatchPipeline(BatchPipeline):
    """Batch pipeline with checkpointing for recovery."""
    
    def __init__(self, config: BatchConfig, checkpoint_path: str):
        super().__init__(config)
        self.checkpoint_manager = CheckpointManager(checkpoint_path)
    
    def run_with_checkpoints(
        self,
        reader,
        writer,
        process_fn,
        checkpoint_interval: int = 1000,
    ):
        """Run batch with periodic checkpoints."""
        
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load()
        
        if checkpoint:
            print(f"Resuming from checkpoint: {checkpoint['last_processed_id']}")
            start_offset = checkpoint["processed_count"]
        else:
            start_offset = 0
        
        processed = start_offset
        errors = []
        
        for batch in reader.read_batches(
            batch_size=self.config.batch_size,
            offset=start_offset,
        ):
            try:
                # Process batch
                results = self._process_with_retry(batch, process_fn)
                
                # Write results
                writer.write_batch(results)
                
                processed += len(batch)
                
                # Checkpoint periodically
                if processed % checkpoint_interval == 0:
                    self.checkpoint_manager.save({
                        "processed_count": processed,
                        "last_processed_id": batch[-1].get("order_id"),
                        "timestamp": datetime.now().isoformat(),
                    })
                    print(f"Checkpoint saved at {processed} records")
                
            except Exception as e:
                errors.append({
                    "batch_start": batch[0].get("order_id") if batch else None,
                    "error": str(e),
                })
                
                if len(errors) > self.config.max_retries:
                    raise RuntimeError(f"Too many batch failures: {errors}")
        
        # Clear checkpoint on success
        self.checkpoint_manager.clear()
        
        return {
            "total_processed": processed,
            "errors": errors,
        }


class CheckpointManager:
    """Manage batch processing checkpoints."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
    
    def save(self, state: dict):
        """Save checkpoint state."""
        
        with open(self.checkpoint_path, "w") as f:
            json.dump(state, f)
    
    def load(self) -> Optional[dict]:
        """Load checkpoint state if exists."""
        
        if not self.checkpoint_path.exists():
            return None
        
        with open(self.checkpoint_path) as f:
            return json.load(f)
    
    def clear(self):
        """Clear checkpoint after successful completion."""
        
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
```

### Phase 5: Progress Monitoring and Alerting

```python
import threading
from datetime import datetime, timedelta

class BatchMonitor:
    """Monitor batch job progress and alert on SLA risk."""
    
    def __init__(
        self,
        sla_deadline: datetime,
        expected_throughput: float,  # records per second
        alert_callback: callable,
    ):
        self.sla_deadline = sla_deadline
        self.expected_throughput = expected_throughput
        self.alert_callback = alert_callback
        
        self._start_time = None
        self._processed_count = 0
        self._lock = threading.Lock()
    
    def start(self, total_records: int):
        """Start monitoring."""
        
        self._start_time = datetime.now()
        self._total_records = total_records
        self._processed_count = 0
    
    def update(self, processed_count: int):
        """Update progress."""
        
        with self._lock:
            self._processed_count = processed_count
            self._check_sla_risk()
    
    def _check_sla_risk(self):
        """Check if we're at risk of missing SLA."""
        
        elapsed = (datetime.now() - self._start_time).total_seconds()
        remaining = self._total_records - self._processed_count
        
        if self._processed_count > 0:
            current_throughput = self._processed_count / elapsed
            estimated_remaining_seconds = remaining / current_throughput
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
            
            # Alert if we might miss SLA
            if estimated_completion > self.sla_deadline:
                buffer_minutes = (self.sla_deadline - estimated_completion).total_seconds() / 60
                
                self.alert_callback({
                    "type": "SLA_AT_RISK",
                    "message": f"Batch job may miss SLA by {abs(buffer_minutes):.0f} minutes",
                    "current_throughput": current_throughput,
                    "required_throughput": remaining / (self.sla_deadline - datetime.now()).total_seconds(),
                    "processed": self._processed_count,
                    "remaining": remaining,
                    "estimated_completion": estimated_completion.isoformat(),
                    "sla_deadline": self.sla_deadline.isoformat(),
                })
    
    def get_status(self) -> dict:
        """Get current status."""
        
        with self._lock:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            throughput = self._processed_count / elapsed if elapsed > 0 else 0
            
            return {
                "processed": self._processed_count,
                "total": self._total_records,
                "progress_pct": (self._processed_count / self._total_records) * 100,
                "elapsed_seconds": elapsed,
                "throughput_per_second": throughput,
                "estimated_remaining_seconds": (self._total_records - self._processed_count) / throughput if throughput > 0 else None,
            }


# Integration with pipeline
def run_monitored_batch():
    """Run batch with SLA monitoring."""
    
    # SLA: Complete by 6 AM
    sla_deadline = datetime.now().replace(hour=6, minute=0, second=0)
    
    monitor = BatchMonitor(
        sla_deadline=sla_deadline,
        expected_throughput=200,  # 200 records/second
        alert_callback=send_slack_alert,
    )
    
    # Count total records
    total_records = count_orders_to_score()
    monitor.start(total_records)
    
    # Run pipeline with progress updates
    pipeline = ResilientBatchPipeline(config, checkpoint_path="./checkpoints/batch")
    
    def process_with_monitoring(batch):
        results = score_batch(batch)
        monitor.update(monitor._processed_count + len(batch))
        return results
    
    result = pipeline.run_with_checkpoints(
        reader=reader,
        writer=writer,
        process_fn=process_with_monitoring,
    )
    
    return {
        **result,
        "sla_met": datetime.now() < sla_deadline,
        "final_status": monitor.get_status(),
    }
```

### Phase 6: Auto-Scaling for Burst Capacity

```python
from google.cloud import run_v2

class AutoScalingBatchRunner:
    """Scale Cloud Run instances for batch processing."""
    
    def __init__(self, project_id: str, region: str, service_name: str):
        self.project_id = project_id
        self.region = region
        self.service_name = service_name
        self.client = run_v2.ServicesClient()
    
    def scale_for_batch(self, target_instances: int):
        """Temporarily increase instance count for batch processing."""
        
        service_path = f"projects/{self.project_id}/locations/{self.region}/services/{self.service_name}"
        
        # Get current service config
        service = self.client.get_service(name=service_path)
        
        # Store original scaling for restoration
        original_min = service.template.scaling.min_instance_count
        original_max = service.template.scaling.max_instance_count
        
        # Update scaling
        service.template.scaling.min_instance_count = target_instances
        service.template.scaling.max_instance_count = max(target_instances, original_max)
        
        self.client.update_service(service=service)
        
        return {
            "original_min": original_min,
            "original_max": original_max,
            "new_min": target_instances,
        }
    
    def restore_scaling(self, original_config: dict):
        """Restore original scaling configuration."""
        
        service_path = f"projects/{self.project_id}/locations/{self.region}/services/{self.service_name}"
        
        service = self.client.get_service(name=service_path)
        service.template.scaling.min_instance_count = original_config["original_min"]
        service.template.scaling.max_instance_count = original_config["original_max"]
        
        self.client.update_service(service=service)


# Usage in batch job
def run_batch_with_autoscaling():
    """Run batch with temporary scale-up."""
    
    scaler = AutoScalingBatchRunner(
        project_id="my-project",
        region="us-central1",
        service_name="delay-risk-api",
    )
    
    # Scale up for batch processing
    original = scaler.scale_for_batch(target_instances=10)
    
    try:
        # Wait for instances to warm up
        time.sleep(60)
        
        # Run batch
        result = run_monitored_batch()
        
    finally:
        # Always restore scaling
        scaler.restore_scaling(original)
    
    return result
```

---

## Results

### Before Implementation

| Metric | Value |
|--------|-------|
| Processing approach | Sequential, single-threaded |
| Throughput | 4.5 orders/second |
| Time for 45K orders | 2h 45m |
| Time for 53K orders | 7h 15m (SLA miss) |
| Recovery from failure | Full restart |

### After Implementation

| Metric | Value |
|--------|-------|
| Processing approach | Parallel, batched |
| Throughput | 180 orders/second |
| Time for 45K orders | 4 minutes |
| Time for 53K orders | 5 minutes |
| Recovery from failure | Resume from checkpoint |

### SLA Performance

| Month | Orders | Duration | SLA Met |
|-------|--------|----------|---------|
| Before optimization | 53,000 | 7h 15m | ❌ |
| After optimization | 53,000 | 5m | ✅ |
| After optimization | 75,000 | 7m | ✅ |
| After optimization | 100,000 | 9m | ✅ |

### Cost Impact

| Metric | Before | After |
|--------|--------|-------|
| Compute time | 7+ hours | 10 minutes |
| Instance hours | 7 | 1.5 (parallel) |
| Cloud Run cost | $15/night | $2/night |
| Planning delay incidents | 3/month | 0/month |

---

## Key Learnings

1. **Batch endpoints are essential** — Single-record APIs don't scale for bulk operations
2. **Parallelism is critical** — 8 threads × 100 batch size = 40x throughput improvement
3. **Checkpoints enable recovery** — Don't restart from zero on transient failures
4. **Monitor against SLA** — Alert before missing deadline, not after
5. **Auto-scale for bursts** — Temporarily increase capacity for predictable batch windows

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Batch Processing Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Reader     │────▶│   Pipeline   │────▶│   Writer     │    │
│  │ (GCS/BQ/CSV) │     │  (Parallel)  │     │ (GCS/BQ/CSV) │    │
│  └──────────────┘     └──────┬───────┘     └──────────────┘    │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │                   │                        │
│              ┌─────▼─────┐       ┌─────▼─────┐                  │
│              │  Worker 1 │  ...  │  Worker N │                  │
│              │  (Batch)  │       │  (Batch)  │                  │
│              └─────┬─────┘       └─────┬─────┘                  │
│                    │                   │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │   Batch API       │                        │
│                    │  /batch/predict   │                        │
│                    └───────────────────┘                        │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Checkpoint  │     │   Monitor    │     │  Auto-Scale  │    │
│  │   Manager    │     │  (SLA Alert) │     │   Manager    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Related Modules

- `batch/pipeline.py` — Core batch processing orchestration
- `batch/io/readers.py` — Input readers (CSV, GCS, BigQuery)
- `batch/io/writers.py` — Output writers (CSV, GCS, BigQuery)
