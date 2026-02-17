# Pathway-Kafka Integration Guide

## Overview
This document describes the integration between Pathway and Kafka for the ICU Digital Twin streaming pipeline.

## Integration Architecture

```
Vital Simulator → Kafka (vitals topic) → Pathway Engine → Kafka (vitals_enriched topic) → ML Service
```

## Key Components

### 1. Kafka Configuration (`app/kafka_config.py`)

The Kafka connector handles:
- Reading streaming data from the `vitals` Kafka topic
- Writing enriched data to the `vitals_enriched` topic
- Connection management and health checks

**Schema Definition:**
```python
class VitalSignsSchema(pw.Schema):
    """Pathway schema for vital signs data"""
    patient_id: str
    timestamp: str
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    respiratory_rate: float
    temperature: float
    shock_index: float
    state: str
    event_type: Optional[str]
```

**Input Stream:**
```python
input_stream = pw.io.kafka.read(
    rdkafka_settings=self.get_consumer_config(),
    topic=self.input_topic,
    schema=VitalSignsSchema,
    format="json",
    autocommit_duration_ms=1000,
)
```

**Output Stream:**
```python
pw.io.kafka.write(
    table=enriched_stream,
    rdkafka_settings=self.get_producer_config(),
    topic=self.output_topic,
    format="json",
)
```

### 2. Risk Engine (`app/risk_engine.py`)

The risk engine performs real-time risk analysis using Pathway's streaming operations:

**Rolling Averages:**
```python
def calculate_rolling_averages(self, vitals_stream: pw.Table) -> pw.Table:
    grouped = vitals_stream.groupby(pw.this.patient_id).reduce(
        patient_id=pw.this.patient_id,
        rolling_hr=pw.reducers.avg(pw.this.heart_rate),
        rolling_spo2=pw.reducers.avg(pw.this.spo2),
        rolling_sbp=pw.reducers.avg(pw.this.systolic_bp),
        record_count=pw.reducers.count(),
    )
    return grouped
```

**Trend Analysis:**
- Joins current vitals with rolling averages
- Calculates trends (current - rolling average)
- Applies risk scoring and anomaly detection

**Risk Calculation:**
- Uses weighted components (shock index, HR trend, SBP trend, SpO2)
- Applies User-Defined Functions (UDFs) via `pw.apply()`
- Normalizes values to [0, 1] range

### 3. Main Stream Processing (`app/main.py`)

**Pipeline Flow:**
1. Validate input data (range checks)
2. Calculate rolling averages (groupby patient_id)
3. Calculate trends (join current with rolling)
4. Apply risk scoring and anomaly detection
5. Write enriched data to output Kafka topic

## Configuration

Environment variables (set via docker-compose.yml):

```yaml
KAFKA_BOOTSTRAP_SERVERS: kafka:9092
KAFKA_INPUT_TOPIC: vitals
KAFKA_OUTPUT_TOPIC: vitals_enriched
KAFKA_CONSUMER_GROUP: vitalx-pathway-engine
```

## Data Flow

### Input (from vital-simulator):
```json
{
  "patient_id": "patient_1",
  "timestamp": "2026-02-14T08:46:00.000Z",
  "heart_rate": 75.0,
  "systolic_bp": 120.0,
  "diastolic_bp": 80.0,
  "spo2": 98.0,
  "respiratory_rate": 16.0,
  "temperature": 36.5,
  "shock_index": 0.625,
  "state": "STABLE",
  "event_type": null
}
```

### Output (enriched with risk analysis):
```json
{
  "patient_id": "patient_1",
  "timestamp": "2026-02-14T08:46:00.000Z",
  ...all original fields...,
  "rolling_hr": 74.5,
  "rolling_spo2": 97.8,
  "rolling_sbp": 118.0,
  "hr_trend": 0.5,
  "sbp_trend": 2.0,
  "computed_risk": 0.15,
  "anomaly_flag": false
}
```

## Recent Integration Fixes

### Issue 1: Schema Recognition
**Problem:** `ValueError: Schema must be specified.`

**Solution:** Ensure schema parameter comes before format parameter in `pw.io.kafka.read()`:
```python
# Correct order:
input_stream = pw.io.kafka.read(
    rdkafka_settings=config,
    topic=topic,
    schema=VitalSignsSchema,  # Schema first
    format="json",            # Then format
    autocommit_duration_ms=1000,
)
```

### Issue 2: Windowing Complexity
**Problem:** Temporal windowing was overly complex for the use case

**Solution:** Simplified to stateful groupby aggregation:
```python
# Instead of temporal.sliding windows, use simpler groupby
grouped = vitals_stream.groupby(pw.this.patient_id).reduce(...)
```

### Issue 3: Join Operations
**Problem:** Join syntax and parameter requirements

**Solution:** Simplified join without explicit JoinMode:
```python
trends = current_vitals.join(
    rolling_averages,
    current_vitals.patient_id == rolling_averages.patient_id,
).select(...)
```

## Testing the Integration

### 1. Build the container:
```bash
docker compose build pathway-engine --no-cache
```

### 2. Start the services:
```bash
docker compose up -d
```

### 3. Check logs:
```bash
docker compose logs pathway-engine -f
```

### 4. Verify data flow:
```bash
# Check vitals topic
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic vitals \
  --from-beginning

# Check enriched output topic
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic vitals_enriched \
  --from-beginning
```

## Dependencies

- pathway>=0.8.0 - Stream processing framework
- confluent-kafka>=2.3.0 - Kafka client
- python-dotenv>=1.0.0 - Environment configuration
- pydantic-settings>=2.0.0 - Settings management

## Troubleshooting

### Container keeps restarting
- Check logs: `docker compose logs pathway-engine`
- Verify Kafka is running: `docker compose ps`
- Check schema definition matches input data

### No data in output topic
- Verify vital-simulator is producing data
- Check consumer group offsets
- Review filter conditions in validation

### High CPU usage
- Monitor Pathway metrics via HTTP endpoint (port 8080)
- Adjust window durations in settings
- Review reducer operations

## Next Steps

1. **Performance Optimization**: Add proper temporal windowing for time-based aggregations
2. **Monitoring**: Integrate with Prometheus/Grafana
3. **Alerting**: Add real-time alerts for high-risk patients
4. **Testing**: Add unit tests for risk calculations and integration tests for Kafka connectivity
