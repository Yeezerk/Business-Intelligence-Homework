import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

# Check available packages
packages = {}
for pkg in ["torch", "shap", "xgboost", "shapiq", "sklearn", "matplotlib", "pandas"]:
    try:
        mod = __import__(pkg)
        packages[pkg] = getattr(mod, "__version__", "installed")
    except ImportError:
        packages[pkg] = "NOT INSTALLED"

print("Package check:")
for k, v in packages.items():
    print(f"  {k}: {v}")

# Check if data exists
from config import PROCESSED_DIR, MODEL_DIR
print(f"\nProcessed data exists: {os.path.exists(os.path.join(PROCESSED_DIR, 'test.pt'))}")
print(f"Models exist: {os.path.exists(os.path.join(MODEL_DIR, 'xgb_baseline.json'))}")
