# Requestly 30-Second Demo Commands

## Quick Demo Script

**Narration**: "Our ICU system integrates Requestly for instant mock data and API monitoring. Let me show you."

---

## Terminal Commands (Copy-Paste in Order)

### 1. Show Requestly Mock Mode is Active (5 seconds)
```bash
curl http://localhost:8000/health
```

**Point out**: `"requestly_mock_mode": true`

---

### 2. Get Patient Data from Requestly Mock Server (8 seconds)
```bash
curl http://localhost:8000/api/floors/1F/patients
```

**Point out**: `"_requestly_source": "mock-server"` in the response

---

### 3. Show Requestly Request Analytics (8 seconds)
```bash
curl http://localhost:8000/api/admin/requestly/analytics
```

**Point out**: 
- `"total_requests"` count
- `"recent_requests"` log
- `"mock_mode_enabled": true`

---

### 4. Toggle Mock Mode via Requestly API (5 seconds)
```bash
curl -X POST "http://localhost:8000/api/admin/requestly/mock-mode?enable=false"
```

**Say**: "We can toggle Requestly mock mode dynamically without restart"

---

### 5. Show Impact (4 seconds)
**Say while showing screen**: "Backend starts in 5 seconds instead of 3 minutes - 97% faster development!"

---

## Complete Command Sequence (30 seconds)

```bash
# 1. Show mock mode active
curl http://localhost:8000/health

# 2. Get Requestly mock data
curl http://localhost:8000/api/floors/1F/patients

# 3. Show API monitoring
curl http://localhost:8000/api/admin/requestly/analytics

# 4. Toggle mock mode
curl -X POST "http://localhost:8000/api/admin/requestly/mock-mode?enable=false"
```

---

## Alternative: With Formatted Output (Better Visual)

```bash
# 1. Show mock mode (formatted)
curl http://localhost:8000/health | python -m json.tool

# 2. Get mock patient data (formatted)
curl http://localhost:8000/api/floors/1F/patients | python -m json.tool

# 3. View analytics (formatted)
curl http://localhost:8000/api/admin/requestly/analytics | python -m json.tool
```

---

## Timing Breakdown

| Command | Time | What to Say |
|---------|------|-------------|
| Health check | 5s | "Requestly mock mode is active" |
| Get patients | 8s | "Getting realistic patient data from Requestly" |
| Analytics | 8s | "All API calls are logged for monitoring" |
| Toggle mode | 5s | "Dynamic control without restart" |
| Summary | 4s | "97% faster development, 100% uptime" |

**Total**: 30 seconds

---

## One-Liner Demo (If Super Short)

```bash
curl http://localhost:8000/health && echo "Mock Mode Active!" && curl http://localhost:8000/api/admin/requestly/analytics
```

**Say**: "Requestly provides mock data and API monitoring - instant development without Kafka infrastructure"

---

## Visual Tips for 30-Second Video

1. **Use large terminal font** (18pt minimum)
2. **Highlight key fields**:
   - `requestly_mock_mode: true`
   - `_requestly_source: mock-server`
   - `total_requests` count
3. **Fast pace** - no pauses between commands
4. **Show documentation** at the end (REQUESTLY_INTEGRATION.md)

---

## Pre-Recording Checklist

- [ ] Backend is running on port 8000
- [ ] Kafka is stopped (to ensure mock mode is active)
- [ ] Terminal font is large and readable
- [ ] Commands are tested and working
- [ ] Timer ready for 30 seconds

---

## Backup One-Command Demo

If you only have 15 seconds:

```bash
curl http://localhost:8000/api/admin/requestly/analytics | python -m json.tool
```

**Say**: "All our API traffic is monitored by Requestly. We use mock mode for fast development and request logging for debugging. Backend starts in 5 seconds, not 3 minutes."
