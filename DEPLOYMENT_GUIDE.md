# Deployment Guide for SentiTrade Web App

## ⚠️ Important: Vercel Limitations

**Vercel is NOT recommended for this application** due to:
- **Size limits**: Models (2-5GB) exceed Vercel's 50MB function limit
- **Timeout limits**: Model loading takes longer than 10-60s timeout
- **Cold starts**: Models must reload on each request

## ✅ Recommended Deployment Platforms

### Option 1: Railway (Easiest - Recommended)

**Why Railway:**
- ✅ No size limits
- ✅ Persistent storage
- ✅ Easy setup
- ✅ Free tier available
- ✅ Supports large ML models

**Steps:**

1. **Install Railway CLI:**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login:**
   ```bash
   railway login
   ```

3. **Initialize project:**
   ```bash
   cd "web app"
   railway init
   ```

4. **Set environment variables:**
   ```bash
   railway variables set NEWS_API_KEY=your_key
   railway variables set FINNHUB_API_KEY=your_key
   railway variables set FMP_API_KEY=your_key
   railway variables set SECRET_KEY=your_secret_key
   railway variables set FLASK_ENV=production
   ```

5. **Deploy:**
   ```bash
   railway up
   ```

6. **Create `railway.json` (optional):**
   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "gunicorn -w 4 -b 0.0.0.0:$PORT wsgi:app",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```

---

### Option 2: Render (Good for Flask)

**Why Render:**
- ✅ Free tier available
- ✅ Supports large files
- ✅ Easy Flask deployment
- ✅ Persistent disks

**Steps:**

1. **Create `render.yaml`:**
   ```yaml
   services:
     - type: web
       name: sentitrade
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: gunicorn -w 4 -b 0.0.0.0:$PORT wsgi:app
       envVars:
         - key: NEWS_API_KEY
           sync: false
         - key: FINNHUB_API_KEY
           sync: false
         - key: FMP_API_KEY
           sync: false
         - key: SECRET_KEY
           generateValue: true
         - key: FLASK_ENV
           value: production
   ```

2. **Deploy:**
   - Push to GitHub
   - Connect repo to Render
   - Render will auto-detect `render.yaml`

---

### Option 3: Fly.io (Container-based)

**Why Fly.io:**
- ✅ Docker-based (full control)
- ✅ Global edge deployment
- ✅ Good for ML workloads

**Steps:**

1. **Create `Dockerfile`:**
   ```dockerfile
   FROM python:3.12-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application
   COPY . .

   # Expose port
   EXPOSE 8080

   # Run with gunicorn
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "wsgi:app"]
   ```

2. **Create `fly.toml`:**
   ```toml
   app = "sentitrade"
   primary_region = "iad"

   [build]

   [http_service]
     internal_port = 8080
     force_https = true
     auto_stop_machines = false
     auto_start_machines = true

   [[vm]]
     memory_mb = 4096
   ```

3. **Deploy:**
   ```bash
   flyctl launch
   flyctl secrets set NEWS_API_KEY=your_key
   flyctl secrets set FINNHUB_API_KEY=your_key
   flyctl secrets set FMP_API_KEY=your_key
   flyctl secrets set SECRET_KEY=your_secret_key
   flyctl deploy
   ```

---

### Option 4: AWS/GCP/Azure (Enterprise)

For production at scale, consider:
- **AWS**: EC2, ECS, or Elastic Beanstalk
- **GCP**: Cloud Run or Compute Engine
- **Azure**: App Service or Container Instances

---

## 📋 Pre-Deployment Checklist

### 1. Environment Variables
Ensure all required variables are set:
- `NEWS_API_KEY`
- `FINNHUB_API_KEY`
- `FMP_API_KEY`
- `SECRET_KEY` (generate a strong secret)
- `FLASK_ENV=production`

### 2. Model Files
- ✅ Models copied to `web app/models/`
- ✅ TFT config in `web app/data/tft/`
- ✅ Source code in `web app/src/`
- ✅ Configs in `web app/configs/`

### 3. Dependencies
- ✅ `requirements.txt` is up to date
- ✅ All packages are listed

### 4. Security
- ✅ `.env` is in `.gitignore`
- ✅ `SECRET_KEY` is strong and unique
- ✅ API keys are not hardcoded

### 5. Production Settings
- ✅ `DEBUG = False` in production config
- ✅ `SECRET_KEY` is set
- ✅ Gunicorn configured properly

---

## 🚀 Quick Start: Railway (Recommended)

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Initialize
cd "web app"
railway init

# 4. Set environment variables
railway variables set NEWS_API_KEY=your_key
railway variables set FINNHUB_API_KEY=your_key
railway variables set FMP_API_KEY=your_key
railway variables set SECRET_KEY=$(openssl rand -hex 32)
railway variables set FLASK_ENV=production

# 5. Deploy
railway up
```

---

## 🔧 Troubleshooting

### Models Not Loading
- Check file paths in `config.py`
- Verify models are in correct directories
- Check file permissions

### Timeout Errors
- Increase timeout in platform settings
- Use persistent storage for models
- Consider model caching

### Memory Issues
- Increase memory allocation
- Use model quantization
- Consider model serving API

---

## 📊 Platform Comparison

| Platform | Free Tier | Size Limit | Timeout | Best For |
|----------|-----------|------------|---------|----------|
| **Railway** | ✅ Yes | Unlimited | Unlimited | ML Apps |
| **Render** | ✅ Yes | Unlimited | Unlimited | Flask Apps |
| **Fly.io** | ✅ Yes | Unlimited | Unlimited | Containers |
| **Vercel** | ✅ Yes | 50MB | 10-60s | Static/API |
| **AWS** | ❌ No | Unlimited | Unlimited | Enterprise |

---

## 💡 Recommendation

**Use Railway** - It's the easiest and best suited for ML applications with large models.

