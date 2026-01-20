# Path Updates After Copying Dependencies

After running `copy_dependencies.py`, the following files have been updated to use local paths instead of referencing "model training":

## ✅ Files Updated

### 1. `config.py`
- **Changed:** `PROJECT_ROOT = BASE_DIR.parent / "model training"` 
- **To:** `PROJECT_ROOT = BASE_DIR`
- **Impact:** All model paths, results paths, and .env file now load from local `web app/` directory

### 2. `app/services/trading_service.py`
- **Changed:** `project_root = Path(__file__).parent.parent.parent.parent / "model training"`
- **To:** `project_root = Path(__file__).parent.parent.parent` (web app directory)
- **Impact:** `src/` imports now work from local `web app/src/` directory

### 3. `.gitignore`
- **Changed:** Was ignoring all `models/` and `*.ckpt`, `*.pkl` files
- **To:** Now allows production models to be tracked:
  - `models/qlora_5fold/` - QLora Llama-2 model
  - `models/tft_checkpoints/` - TFT checkpoints
  - `models/sentiment_production/` - Sentiment models
  - `data/tft/` - TFT config files
- **Still ignores:** Training artifacts (ablation_studies, baselines, etc.)

## 📁 Directory Structure After Copy

```
web app/
├── src/                          # Copied from model training/src/
│   ├── data/
│   │   ├── collectors.py
│   │   └── preprocessors.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
├── models/                       # Copied models
│   ├── qlora_5fold/
│   │   └── fold_0_final/
│   └── tft_checkpoints/
│       └── tft-epoch=37-val_loss=0.0237.ckpt
├── data/
│   └── tft/
│       ├── tft_config.json
│       └── scalers.pkl
├── configs/                      # Copied from model training/configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── .env                          # Copied from model training/.env
├── final_results_summary.json    # Optional
└── results/                      # Optional
    └── qlora_5fold_final_results.json
```

## ✅ Verification

After copying and updating paths, verify:

1. **Check paths are correct:**
   ```python
   # In config.py, these should now point to local directories:
   QLORA_MODEL_PATH = BASE_DIR / "models" / "qlora_5fold" / "fold_0_final"
   TFT_MODEL_PATH = BASE_DIR / "models" / "tft_checkpoints" / "tft-epoch=37-val_loss=0.0237.ckpt"
   ```

2. **Test the app:**
   ```bash
   cd "web app"
   python run.py
   ```

3. **Verify endpoints:**
   - `http://localhost:5000/analyze` - Should load sentiment model
   - `http://localhost:5000/signals` - Should load TFT model and data collectors

4. **Check Git status:**
   ```bash
   git status
   # Should show models/, src/, data/, configs/ as new files to add
   ```

## ⚠️ Important Notes

1. **Model File Sizes:** Model files are large (500MB-2GB). Consider:
   - Using Git LFS for large files
   - Or hosting models separately and downloading on deployment

2. **Environment Variables:** The `.env` file is ignored by Git (as it should be). Make sure to:
   - Set up environment variables in your deployment platform
   - Or use a secrets management service

3. **Config Files:** The `configs/*.yaml` files may reference paths that need updating if they're absolute paths.

