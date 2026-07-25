"""
下载数据：Kaggle Credit Card Fraud Detection Dataset 2023。
原始数据集是 50:50 均衡的（人工平衡），后续需下采样到 0.5% 还原真实场景。

优先从 Google Drive 镜像下载（更快），失败则回退到 kagglehub。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from config import RAW_DATA

KAGGLE_PATH = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
GDRIVE_FILE_ID = "1ipjEJoZQMa5AMaaoJ1y3kv988jkC_uKL"
GDRIVE_FILENAME = "creditcard_2023.csv"


def download_gdrive():
    """从 Google Drive 镜像下载（gdown 库）。"""
    import gdown
    os.makedirs(os.path.dirname(RAW_DATA), exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading from Google Drive: {url}")
    gdown.download(url, RAW_DATA, quiet=False)
    print(f"Saved to {RAW_DATA}")


def download_kaggle():
    """从 Kaggle 下载（kagglehub 库，失败回退）。"""
    import kagglehub
    os.makedirs(os.path.dirname(RAW_DATA), exist_ok=True)
    print(f"Downloading from Kaggle: {KAGGLE_PATH} ...")
    dl_path = kagglehub.dataset_download(KAGGLE_PATH)
    print(f"Downloaded to: {dl_path}")
    for fname in os.listdir(dl_path):
        if fname.endswith(".csv"):
            src = os.path.join(dl_path, fname)
            shutil.copy2(src, RAW_DATA)
            print(f"Copied {fname} → {RAW_DATA}")
            return
    raise FileNotFoundError(f"No CSV found in {dl_path}")


def main():
    # 如果已下载则跳过（删除文件可重新下载）
    if os.path.exists(RAW_DATA):
        size_mb = os.path.getsize(RAW_DATA) / 1024 / 1024
        print(f"Data already exists: {RAW_DATA} ({size_mb:.0f} MB)")
        print("Delete it to re-download.")
        return

    try:
        download_gdrive()
    except Exception as e:
        print(f"Google Drive download failed: {e}")
        print("Falling back to Kaggle download...")
        download_kaggle()


if __name__ == "__main__":
    main()
