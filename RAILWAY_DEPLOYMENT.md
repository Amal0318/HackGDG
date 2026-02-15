# 🚀 Railway Deployment - Hackathon Quick Start

**Time needed:** 15-20 minutes  
**Cost:** ~$5 for 2 hours or FREE with Railway trial credits

---

## ⚡ Quick Deploy (Easiest Method)

### 1. Install Railway CLI
```bash
# Windows (PowerShell)
npm install -g @railway/cli

# Or download from: https://railway.app/cli
```

### 2. Login & Initialize
```bash
railway login
cd d:\Programs\HackGDG_Final
railway init
```

### 3. Deploy All Services
```bash
# Railway will auto-detect docker-compose.yml and deploy all services
railway up
```

### 4. Get Your URLs
```bash
# Get backend URL
railway open backend-api

# Get frontend URL  
railway open frontend
```

**That's it!** ✅ Your app is live!

---

## 🎯 Alternative: Railway Dashboard Method

### Step 1: Push to GitHub (if not already)
```bash
git init
git add .
git commit -m "Ready for Railway deployment"
gh repo create HackGDG-ICU --public --source=. --push
```

### Step 2: Deploy via Railway Dashboard
1. Go to https://railway.app
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your `HackGDG-ICU` repository
5. Railway will detect `docker-compose.yml` automatically

### Step 3: Configure Services
Railway will create 7 services automatically. For each service that needs a public URL:

**backend-api:**
- Go to service → Settings → Generate Domain
- Copy the URL (e.g., `backend-api.up.railway.app`)

**frontend:**
- Go to service → Settings → Generate Domain
- Add environment variable: `VITE_API_URL` = `https://backend-api.up.railway.app`
- Redeploy frontend

---

## 🔧 Important Configuration

### Frontend API Connection
Update [frontend/src/services/api.ts](frontend/src/services/api.ts):

```typescript
// Change this line to use Railway backend URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### Backend CORS (if needed)
Update [icu-system/backend-api/main.py](icu-system/backend-api/main.py) to allow Railway domains:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon - allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 Service Boot Order & Health Check

Services will start in this order:
1. ✅ Zookeeper (30s)
2. ✅ Kafka (45s)
3. ✅ Vital Simulator (20s)
4. ✅ Pathway Engine (30s)
5. ✅ ML Service (60s - loads model)
6. ✅ Backend API (30s)
7. ✅ Frontend (20s)

**Total startup time:** ~4-5 minutes

Check if services are ready:
```bash
# Health check
curl https://[your-backend-url].railway.app/health

# Get patients
curl https://[your-backend-url].railway.app/api/floors/1F/patients
```

---

## 💰 Cost Breakdown (Hourly)

| Service | RAM | Cost/hour |
|---------|-----|-----------|
| Zookeeper | 512MB | $0.50 |
| Kafka | 1GB | $1.00 |
| Vital Simulator | 256MB | $0.25 |
| Pathway Engine | 512MB | $0.50 |
| ML Service | 1GB | $1.00 |
| Backend API | 512MB | $0.50 |
| Frontend | 256MB | $0.25 |
| **TOTAL** | **4GB** | **~$4/hr** |

**For 2-hour demo:** ~$8 total
**Railway gives $5 free credits** → Net cost: ~$3

---

## 🎬 During Your Demo

### Test the deployment:
```bash
# Set your Railway backend URL
set BACKEND_URL=https://your-backend.railway.app

# 1. Health check
curl %BACKEND_URL%/health

# 2. Get patients
curl %BACKEND_URL%/api/floors/1F/patients

# 3. Check WebSocket (if applicable)
# Open browser console at your frontend URL
# Look for WebSocket connection messages
```

### Access monitoring:
```bash
# View logs in real-time
railway logs backend-api --follow
railway logs ml-service --follow
```

---

## 🛑 After Hackathon - Save Money!

Delete the project to stop charges:
```bash
railway down

# Or via dashboard: Project Settings → Delete Project
```

---

## 🚨 Troubleshooting

### Services fail to connect?
- Check Railway internal networking uses `.railway.internal` domains
- Verify environment variables are set correctly

### Frontend can't reach backend?
- Make sure you generated a public domain for backend-api
- Check VITE_API_URL environment variable in frontend
- Rebuild frontend after adding env var

### Kafka not starting?
- Increase Kafka service memory to 1.5GB in Railway dashboard
- Check Zookeeper is running first

### ML model loading slow?
- First request takes 60s (model loading)
- Subsequent requests are fast
- Pre-warm: `curl your-backend/api/predict` before demo

---

## ✨ Pro Tips for Hackathon

1. **Deploy 30 min before your presentation** - gives buffer for issues
2. **Test all features once deployed** - don't assume it works
3. **Keep Railway dashboard open** - monitor logs during demo
4. **Have backup localhost running** - in case Railway has issues
5. **Share frontend URL with judges** - let them explore

---

## 📱 What Judges Will See

**Frontend URL:** `https://icu-frontend-production.up.railway.app`
- Full dashboard
- Real-time patient monitoring
- Risk predictions
- Live vitals updates

**Backend URL:** `https://icu-backend-production.up.railway.app`
- REST API endpoints
- WebSocket for real-time data
- Health monitoring

---

## Need Help?

Railway Discord: https://discord.gg/railway  
Railway Docs: https://docs.railway.app

**Good luck with your hackathon! 🎉**
