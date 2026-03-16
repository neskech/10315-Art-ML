import os
import zipfile


from data_generation.download_from_google import download_from_google

FILE_ID = ''
DATA_PATH = '../data/'


def _download_zip_to_path(path: str):
    download_from_google()


def main():
    if os.path.exists(DATA_PATH):
        os.rmdir(DATA_PATH)

    poses_path = os.path.join(DATA_PATH, 'poses/')
    os.makedirs(poses_path, exist_ok=False)

    zip_path = os.path.join(DATA_PATH, 'data.zip')
    _download_zip_to_path(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(poses_path)
    os.remove(zip_path)
