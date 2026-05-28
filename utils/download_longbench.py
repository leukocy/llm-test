"""
LongBench under载脚本 (ZIP版)
under载 data.zip 并解压
"""

import os
import shutil
import zipfile

from huggingface_hub import hf_hub_download

DATASETS_DIR = "datasets/longbench"

def download_and_extract():
    print("🚀 开始under载 LongBench data.zip...")
    os.makedirs(DATASETS_DIR, exist_ok=True)

    try:
        # under载 data.zip
        zip_path = hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            local_dir=DATASETS_DIR
        )
        print(f"✅ Download complete: {zip_path}")

        # 解压
        print("📦 currently解压...")
        with zipfile.ZipFile(os.path.join(DATASETS_DIR, "data.zip"), 'r') as zip_ref:
            zip_ref.extractall(DATASETS_DIR)

        # 整理目录结构
        # 解压后通常willhas data/ 目录
        extracted_data_dir = os.path.join(DATASETS_DIR, "data")
        if os.path.exists(extracted_data_dir):
            print("🗂️ 整理文件...")
            for filename in os.listdir(extracted_data_dir):
                src = os.path.join(extracted_data_dir, filename)
                dst = os.path.join(DATASETS_DIR, filename)
                if os.path.isfile(src):
                    # if目标存in，先Delete
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)

            # Delete空 data 目录
            os.rmdir(extracted_data_dir)

        # Delete zip 文件 (optional)
        # os.remove(os.path.join(DATASETS_DIR, "data.zip"))

        print("\n✨ LongBench Dataset准备完成！")
        print(f"   位置: {DATASETS_DIR}")
        return True

    except Exception as e:
        print(f"❌ under载or解压失败: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
