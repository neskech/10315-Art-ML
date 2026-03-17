import os
from pathlib import Path
import zipfile
import shutil
import gdown

FILE_ID = "1rO4YwzWQ78anjThH7une-waX86Fkv02q"
CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / 'data'

def main():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

    poses_path = os.path.join(DATA_PATH, "poses/")
    os.makedirs(poses_path, exist_ok=True)

    zip_path = os.path.join(DATA_PATH, "data.zip")

    print(f"Downloading file ID: {FILE_ID}")
    gdown.download(id=FILE_ID, output=zip_path, quiet=False)

    # Sanity check to ensure download succeeded before extracting
    if not os.path.exists(zip_path):
        raise Exception(
            "Download failed! Ensure the Google Drive link is set to 'Anyone with the link can view'."
        )

    # Extract and clean up
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(poses_path)

    os.remove(zip_path)


if __name__ == "__main__":
    print("Downloading data from Google Drive...")
    main()
    print("Finished downloading data from Google Drive!")