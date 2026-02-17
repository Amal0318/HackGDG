# 🚀 Quick Start - Email Alerts with Gemini 2.5 Flash

## What You Need to Do:

### 1️⃣ Get Gemini API Key (FREE - Takes 30 seconds)
```
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key (starts with AIzaSy...)
```

### 2️⃣ Get Gmail App Password (Takes 1 minute)
```
1. Visit: https://myaccount.google.com/apppasswords
2. Sign in to Gmail
3. Select "Mail" + your device
4. Copy the 16-character password
```

### 3️⃣ Create .env File
```bash
cd icu-system
cp .env.example .env
```

### 4️⃣ Edit .env File
```env
GEMINI_API_KEY=AIzaSy...paste_your_key_here
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop
ALERT_EMAIL_TO=doctor-email@hospital.com
```

### 5️⃣ Start Everything
```bash
docker-compose up -d
```

### 6️⃣ Check if Emails Work
```bash
docker logs -f icu-alert-system
```

Look for: `✅ EMAIL SENT to doctor@hospital.com`

---

## ✅ What's Already Done
- ✅ LangChain + Gemini 2.5 Flash integration
- ✅ Email HTML templates
- ✅ Alert system with rate limiting
- ✅ Docker configuration
- ✅ Backend API (http://localhost:8000)
- ✅ Requestly sponsor integration

## 📊 Alert Triggers
- Risk Score > 0.7
- Shock Index > 1.0  
- SpO2 < 90%
- Anomaly detected

## 🔥 Next Steps for Hackathon Demo
1. ⏳ Build frontend dashboard (login + floors + patient cards)
2. ⏳ Deploy to Railway (10-15 hour demo)
3. ⏳ Test complete email flow

## 🏥 System Architecture
- 3 Floors × 8 Patients = 24 Total
- Real-time Kafka streaming
- Pathway engine enrichment
- ML-based risk scoring
- LangChain intelligent alerts
- Multi-channel notifications

---
**Medical AI Monitoring System | Powered by Gemini 2.5 Flash**
