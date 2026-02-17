# 🚀 VitalX - Ultra-Fast Ngrok Deployment (2 Minutes!)

## Step 1: Download Ngrok (30 seconds)

1. Open browser and go to: **https://ngrok.com/download**
2. Click **"Download for Windows"**
3. Extract the ZIP file to `d:\Programs\ngrok` (or anywhere you like)
4. No installation needed - it's a single .exe file!

**Or use this direct link:** https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip

---

## Step 2: Start Your Local Services (1 minute)

Open **new Command Prompt** (as Administrator if possible):

```cmd
cd d:\Programs\HackGDG_Final
docker-compose up -d
```

Wait ~2 minutes for all services to start. Check status:
```cmd
docker-compose ps
```

**Expected output:**
- icu-frontend (port 3000) ✅
- icu-backend-api (port 8000) ✅  
- icu-ml-service (port 8001) ✅
- icu-kafka, icu-zookeeper, etc. ✅

---

## Step 3: Expose Frontend with Ngrok (30 seconds)

Open **another Command Prompt**:

```cmd
cd d:\Programs\ngrok
ngrok http 3000
```

**You'll see something like:**
```
Session Status: online
Forwarding: https://abc123xyz.ngrok-free.app -> http://localhost:3000
```

**Copy that URL!** That's your live demo link! 🎉

---

## Step 4: Share with Judges! ✅

**Frontend URL:** `https://abc123xyz.ngrok-free.app` (from step 3)

This URL is:
- ✅ Publicly accessible
- ✅ HTTPS secured
- ✅ Works on any device
- ✅ Perfect for demo!

---

## Optional: Expose Backend Too (for API testing)

If judges want to test the API directly:

Open **third Command Prompt**:
```cmd
cd d:\Programs\ngrok
ngrok http 8000
```

You'll get another URL like: `https://def456uvw.ngrok-free.app`

---

## 📊 What to Show Judges

1. **Frontend Dashboard:** `https://[your-ngrok-url].ngrok-free.app`
   - Chief/Doctor/Nurse dashboards
   - Real-time patient monitoring
   - Risk predictions
   - Live vitals

2. **API Endpoints** (if they ask):
   ```bash
   # Health check
   curl https://[backend-ngrok-url].ngrok-free.app/health
   
   # Get patients
   curl https://[backend-ngrok-url].ngrok-free.app/api/floors/1F/patients
   ```

---

## 🛑 After Hackathon

Stop services:
```cmd
docker-compose down
```

Stop ngrok: Press `Ctrl+C` in the ngrok terminal

---

## ⚠️ Troubleshooting

**Docker not starting?**
1. Open Docker Desktop
2. Wait for it to say "Docker is running"
3. Try Step 2 again

**Services slow to start?**
- Wait 3-4 minutes for all services
- ML service takes longest (loads AI model)
- Check logs: `docker-compose logs -f`

**Ngrok asking for auth?**
1. Sign up free at https://ngrok.com
2. Get your authtoken
3. Run: `ngrok config add-authtoken YOUR_TOKEN`
4. Try Step 3 again

---

## 💡 Pro Tips

1. **Test before presenting:** Open the ngrok URL on your phone to confirm it works!
2. **Keep terminals open:** Don't close the ngrok window during demo
3. **Have backup:** Keep localhost:3000 ready in case ngrok has issues
4. **Warm up ML:** Make one API call before demo so model is loaded

---

**Total Time:** 2-3 minutes  
**Cost:** FREE! 🎉  
**Reliability:** Perfect for 1-2 hour demo

Good luck with your hackathon! 🚀
