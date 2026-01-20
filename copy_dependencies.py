"""
Script to copy required dependencies from model training to web app
Run this from the project root directory
"""
import shutil
from pathlib import Path
import sys

# Project root (parent of both 'web app' and 'model training')
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_TRAINING_DIR = PROJECT_ROOT / "model training"
WEB_APP_DIR = PROJECT_ROOT / "web app"

def copy_directory(src: Path, dst: Path, description: str):
    """Copy a directory and its contents"""
    if not src.exists():
        print(f" WARNING: {description} not found at {src}")
        return False
    
    print(f"Copying {description}...")
    try:
        if dst.exists():
            print(f"   Removing existing {dst}")
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied to {dst}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def copy_file(src: Path, dst: Path, description: str):
    """Copy a single file"""
    if not src.exists():
        print(f"WARNING: {description} not found at {src}")
        return False
    
    print(f"📄 Copying {description}...")
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"   ✅ Copied to {dst}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("Copying Dependencies from 'model training' to 'web app'")
    print("="*70)
    print()
    
    if not MODEL_TRAINING_DIR.exists():
        print(f"ERROR: 'model training' directory not found at {MODEL_TRAINING_DIR}")
        sys.exit(1)
    
    if not WEB_APP_DIR.exists():
        print(f"ERROR: 'web app' directory not found at {WEB_APP_DIR}")
        sys.exit(1)
    
    # Track what was copied
    copied = []
    failed = []
    
    # 1. Copy source code (src/)
    src_src = MODEL_TRAINING_DIR / "src"
    src_dst = WEB_APP_DIR / "src"
    if copy_directory(src_src, src_dst, "Source code (src/)"):
        copied.append("src/")
    else:
        failed.append("src/")
    
    # 2. Copy QLora model
    qlora_src = MODEL_TRAINING_DIR / "models" / "qlora_5fold" / "fold_0_final"
    qlora_dst = WEB_APP_DIR / "models" / "qlora_5fold" / "fold_0_final"
    if copy_directory(qlora_src, qlora_dst, "QLora Llama-2 model"):
        copied.append("models/qlora_5fold/fold_0_final/")
    else:
        failed.append("models/qlora_5fold/fold_0_final/")
    
    # 3. Copy TFT checkpoint
    tft_checkpoint_src = MODEL_TRAINING_DIR / "models" / "tft_checkpoints" / "tft-epoch=37-val_loss=0.0237.ckpt"
    tft_checkpoint_dst = WEB_APP_DIR / "models" / "tft_checkpoints" / "tft-epoch=37-val_loss=0.0237.ckpt"
    if copy_file(tft_checkpoint_src, tft_checkpoint_dst, "TFT model checkpoint"):
        copied.append("models/tft_checkpoints/tft-epoch=37-val_loss=0.0237.ckpt")
    else:
        failed.append("models/tft_checkpoints/tft-epoch=37-val_loss=0.0237.ckpt")
    
    # 4. Copy TFT patched checkpoint if it exists
    tft_patched_src = MODEL_TRAINING_DIR / "models" / "tft_checkpoints" / "tft-epoch=37-val_loss=0.0237_patched.ckpt"
    if tft_patched_src.exists():
        tft_patched_dst = WEB_APP_DIR / "models" / "tft_checkpoints" / "tft-epoch=37-val_loss=0.0237_patched.ckpt"
        if copy_file(tft_patched_src, tft_patched_dst, "TFT patched checkpoint"):
            copied.append("models/tft_checkpoints/tft-epoch=37-val_loss=0.0237_patched.ckpt")
    
    # 5. Copy TFT configuration files
    tft_config_src = MODEL_TRAINING_DIR / "data" / "tft" / "tft_config.json"
    tft_config_dst = WEB_APP_DIR / "data" / "tft" / "tft_config.json"
    if copy_file(tft_config_src, tft_config_dst, "TFT config (tft_config.json)"):
        copied.append("data/tft/tft_config.json")
    else:
        failed.append("data/tft/tft_config.json")
    
    # 6. Copy TFT scalers (optional but recommended)
    tft_scalers_src = MODEL_TRAINING_DIR / "data" / "tft" / "scalers.pkl"
    tft_scalers_dst = WEB_APP_DIR / "data" / "tft" / "scalers.pkl"
    if tft_scalers_src.exists():
        if copy_file(tft_scalers_src, tft_scalers_dst, "TFT scalers (scalers.pkl)"):
            copied.append("data/tft/scalers.pkl")
    else:
        print("WARNING: scalers.pkl not found (optional file)")
    
    # 7. Copy configs directory
    configs_src = MODEL_TRAINING_DIR / "configs"
    configs_dst = WEB_APP_DIR / "configs"
    if copy_directory(configs_src, configs_dst, "Configuration files (configs/)"):
        copied.append("configs/")
    else:
        failed.append("configs/")
        print("WARNING: Config files are required for data collectors to work")
    
    # 8. Copy .env file
    env_src = MODEL_TRAINING_DIR / ".env"
    env_dst = WEB_APP_DIR / ".env"
    if copy_file(env_src, env_dst, "Environment file (.env)"):
        copied.append(".env")
    else:
        failed.append(".env")
        print("You'll need to create .env manually with your API keys")
    
    # 9. Copy results files (optional)
    print()
    print("Copying results/metrics files (optional)...")
    
    results_files = [
        ("final_results_summary.json", "Final results summary"),
        ("results/qlora_5fold_final_results.json", "QLora results"),
        ("models/test_metrics.json", "TFT test metrics"),
    ]
    
    for rel_path, description in results_files:
        src = MODEL_TRAINING_DIR / rel_path
        dst = WEB_APP_DIR / rel_path
        if src.exists():
            if copy_file(src, dst, description):
                copied.append(rel_path)
        else:
            print(f"{description} not found (optional)")
    
    # Summary
    print()
    print("="*70)
    print("COPY SUMMARY")
    print("="*70)
    print(f"Successfully copied: {len(copied)} items")
    for item in copied:
        print(f"   • {item}")
    
    if failed:
        print()
        print(f"Failed to copy: {len(failed)} items")
        for item in failed:
            print(f"   • {item}")
        print()
        print("WARNING: Some required files were not copied!")
        print("The web app may not work correctly without these files.")
    
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Update 'web app/config.py' to point to local files:")
    print("   Change: PROJECT_ROOT = BASE_DIR.parent / 'model training'")
    print("   To:     PROJECT_ROOT = BASE_DIR")
    print()
    print("2. Verify all files were copied correctly")
    print("3. Test the web app locally:")
    print("   cd 'web app'")
    print("   python run.py")
    print("4. Check that /analyze and /signals endpoints work")
    print("="*70)

if __name__ == "__main__":
    main()

