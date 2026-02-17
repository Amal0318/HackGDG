# 🔧 CORS Error Fix - Backend Restart Required

## Problem
Frontend (http://localhost:3000) can't connect to backend (http://localhost:8000) due to CORS blocking.

## Solution
The backend needs to be restarted to load the updated CORS configuration.

## Steps to Fix (Run in PowerShell or CMD with Docker):

### Option 1: Quick Restart (Recommended)
```bash
cd d:\Programs\HackGDG_Final\icu-system

# Stop and remove old backend container
docker stop icu-backend-api
docker rm icu-backend-api

# Rebuild and restart with new CORS config
docker-compose up -d --build backend-api

# Wait 10 seconds for startup
timeout /t 10

# Test it works
curl http://localhost:8000/
```

### Option 2: Restart All Services
```bash
cd d:\Programs\HackGDG_Final\icu-system
docker-compose down
docker-compose up -d --build
```

### Option 3: Use the Batch Script
```bash
d:\Programs\HackGDG_Final\icu-system\restart_backend.bat
```

## Verify Fix Worked

After restart, test CORS:
```bash
curl -X OPTIONS http://localhost:8000/api/auth/login -H "Origin: http://localhost:3000" -v
```

Should see headers like:
```
< access-control-allow-origin: http://localhost:3000
< access-control-allow-credentials: true
< access-control-allow-methods: *
< access-control-allow-headers: *
```

Then refresh frontend at http://localhost:3000 and login should work!

## What Changed

**File: `backend-api/app/config.py`** (Already updated ✅)
```python
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",  # Frontend dev server
    "http://localhost:5173",  # Vite default port
    "http://localhost:8080",  # Alternative
    "*"  # Allow all for demo
]
```

**File: `backend-api/app/main.py`** (Already updated ✅)  
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Current Issue
Backend container is running OLD code (version 0.1.0).  
After restart, it will run NEW code (version 1.0.0) with proper CORS.

---
**TL;DR**: Open a terminal with Docker, run `restart_backend.bat`, then try login again at http://localhost:3000
