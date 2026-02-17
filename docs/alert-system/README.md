# 🚨 Alert System - LangChain Emergency Notifications

**AI-powered intelligent alerts for high-risk ICU patients**

## 🎯 Features

### ✅ LangChain Integration
- **Smart alert generation** using LLM (Gemini or OpenAI)
- **Contextual analysis** of patient vitals and trends
- **Natural language** medical alerts for doctors
- **Automatic severity classification** (CRITICAL / HIGH / MEDIUM)

### 📡 Real-time Monitoring
- Consumes from Kafka `vitals_enriched` topic
- Monitors 24 patients across 3 ICU floors
- Triggers on high-risk conditions
- Rate limiting to prevent alert spam (5min cooldown)

### 📬 Multi-channel Notifications
- **Console**: Formatted alerts with color coding
- **Webhook**: Slack/Discord integration
- **Backend API**: Alert logging for dashboard
- **Extensible**: Easy to add Email/SMS

### 🧠 Intelligent Triggering
Alerts triggered when:
- Risk score > 0.7
- Shock index > 1.0 (critical)
- SpO2 < 90% (hypoxemia)
- Anomaly detected by ML model
- Patient marked as high-risk

---

## 🚀 Quick Start

### 1. Get API Key (Choose One)

#### Option A: Gemini (Recommended - FREE)
1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key

#### Option B: OpenAI
1. Visit: https://platform.openai.com/api-keys
2. Create new secret key
3. Copy your key (starts with `sk-`)

### 2. Configure Environment

```bash
cd alert-system
cp .env.example .env
# Edit .env and add your API key
```

### 3. Run with Docker Compose

Add to `docker-compose.yml`:
```yaml
  alert-system:
    build:
      context: ./alert-system
    container_name: icu-alert-system
    depends_on:
      kafka:
        condition: service_healthy
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      LLM_PROVIDER: gemini  # or openai
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      # OPENAI_API_KEY: ${OPENAI_API_KEY}
    networks:
      - icu-network
    restart: unless-stopped
```

### 4. Start the System

```bash
# From icu-system directory
docker compose up -d alert-system

# View alerts in real-time
docker logs icu-alert-system -f
```

---

## 📊 Example Alert Output

```
================================================================================
🔴 CRITICAL ALERT - 2026-02-14 23:45:32 UTC
================================================================================
Patient: P2-003 | Floor: 2F
--------------------------------------------------------------------------------
CRITICAL PRIORITY

Patient P2-003 on Floor 2F is experiencing severe hemodynamic instability 
requiring immediate intervention.

Key Findings:
• Shock Index of 1.61 indicates severe hypovolemic or cardiogenic shock
• SpO2 at 89% shows significant hypoxemia below critical threshold
• Heart rate elevated 35% above baseline with positive 35.2 bpm trend
• Blood pressure critically low with negative trend (-18.4 mmHg)

Recommended Action:
Immediate bedside assessment required. Consider fluid resuscitation, 
vasopressor support, and supplemental oxygen. Prepare for potential ICU 
escalation procedures.

Clinical Reasoning:
The combination of elevated shock index, hypoxemia, and deteriorating 
hemodynamics suggests acute decompensation. The 87% risk score from the 
predictive model corroborates clinical findings. Anomaly detection has 
flagged unusual vital sign patterns.
--------------------------------------------------------------------------------
LLM Provider: GEMINI
================================================================================
```

---

## 🎛️ Configuration Options

### Environment Variables

| Variable | Options | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` / `openai` | Choose LLM provider |
| `GEMINI_API_KEY` | String | Your Gemini API key |
| `OPENAI_API_KEY` | String | Your OpenAI API key |
| `KAFKA_BOOTSTRAP_SERVERS` | URL | Kafka connection |
| `ENABLE_WEBHOOK_ALERTS` | `true` / `false` | Enable Slack/Discord |
| `WEBHOOK_URL` | URL | Webhook endpoint |

### Alert Thresholds (in `config.py`)

```python
HIGH_RISK_THRESHOLD = 0.7      # Risk score threshold
CRITICAL_SHOCK_INDEX = 1.0     # Shock index threshold
CRITICAL_SPO2 = 90             # SpO2 threshold
MIN_ALERT_INTERVAL_SECONDS = 300  # Rate limiting
```

---

## 🔗 Slack/Discord Integration

### Slack Setup:
1. Create Slack App: https://api.slack.com/apps
2. Add "Incoming Webhooks" feature
3. Create webhook for your channel
4. Copy webhook URL

### Discord Setup:
1. Go to Server Settings → Integrations
2. Create Webhook
3. Copy webhook URL

### Configure:
```bash
ENABLE_WEBHOOK_ALERTS=true
WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

---

## 📈 Alert Statistics

View statistics in logs on shutdown:
```
📊 Session Statistics:
   Total alerts sent: 42
   By severity: {'CRITICAL': 8, 'HIGH': 24, 'MEDIUM': 10}
   By floor: {'1F': 12, '2F': 18, '3F': 12}
```

---

## 🧪 Testing Without Kafka

For local testing, create `test_alert.py`:

```python
from app.langchain_agent import MedicalAlertAgent

agent = MedicalAlertAgent()

test_data = {
    "patient_id": "P1-001",
    "floor_id": "1F",
    "state": "unstable",
    "rolling_hr": 145,
    "rolling_sbp": 90,
    "spo2": 89,
    "shock_index": 1.61,
    "hr_trend": 35.2,
    "sbp_trend": -18.4,
    "computed_risk": 0.87,
    "anomaly_flag": 1,
    "is_high_risk": True
}

alert = agent.generate_alert(test_data)
print(alert['alert_message'])
```

---

## 💰 Cost Estimation

### Gemini (Recommended for Demo):
- **FREE tier**: 60 requests/minute
- Cost: **$0.00** for 10-15 hour demo
- Perfect for hackathon!

### OpenAI:
- GPT-3.5-turbo: $0.0005 per alert
- ~100 alerts in demo = **$0.05**
- Negligible cost

---

## 🎯 Demo Tips for Judges

1. **Show live alerts** in terminal during demo
2. **Explain LangChain integration**: "We use LangChain to generate intelligent, context-aware medical alerts"
3. **Highlight multi-channel**: "System can notify via console, Slack, SMS, email"
4. **Demonstrate resilience**: "If LLM fails, falls back to rule-based alerts"
5. **Show alert history**: Point to statistics on shutdown

### Impressive Talking Points:
- ✨ "We integrated LangChain for intelligent alert generation"
- ✨ "Supports both Gemini and OpenAI LLMs"
- ✨ "Rate-limited to prevent alert fatigue"
- ✨ "Natural language alerts for better doctor comprehension"
- ✨ "Multi-channel notification system"

---

## 🏗️ Architecture

```
Kafka (vitals_enriched)
    ↓
Alert System Consumer
    ↓
High-Risk Detection
    ↓
LangChain Agent
    ↓
LLM (Gemini/OpenAI)
    ↓
Smart Alert Generation
    ↓
Multi-Channel Notifications
    ├─ Console
    ├─ Slack/Discord
    └─ Backend API
```

---

## 🐛 Troubleshooting

**No alerts appearing:**
- Check if vital-simulator and pathway-engine are running
- Verify patients are being detected as high-risk
- Check alert rate limiting (5min cooldown)

**LLM errors:**
- Verify API key is correct
- Check API quota/billing
- System will fall back to rule-based alerts

**Kafka connection failed:**
- Ensure Kafka container is healthy
- Check `KAFKA_BOOTSTRAP_SERVERS` environment variable

---

## 🎓 Learning Resources

- LangChain Docs: https://python.langchain.com/docs/get_started/introduction
- Gemini API: https://ai.google.dev/tutorials/python_quickstart
- OpenAI API: https://platform.openai.com/docs/quickstart

---

## 📝 Future Enhancements

- [ ] Email notification support
- [ ] SMS via Twilio
- [ ] Alert dashboard in frontend
- [ ] Historical alert analytics
- [ ] Custom alert templates per role
- [ ] Multi-language support
- [ ] Voice call escalation for CRITICAL alerts

---

## 🏆 Built For Hackathon Success

This alert system demonstrates:
- ✅ Advanced AI/LLM integration
- ✅ Real-world healthcare application
- ✅ Production-ready architecture patterns
- ✅ Intelligent automation layer
- ✅ Scalable, extensible design

---

**Made with ❤️ using LangChain, Gemini/OpenAI, and Kafka**
