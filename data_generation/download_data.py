from pathlib import Path
import zipfile
import gdown

CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / "data"
POSES_PATH = DATA_PATH / "poses"

DATASETS = {
    "pinterest": "1DIg-3zQp5aPkxDQ1CgbqmTyJPCRhxSgp",
    "sports": "1orMqDftzDZdKybAvqST9IEQpwZ04OcaQ",
    "MPII": "1aS7qjZcTi2yjqshToLi4AL8fdhYX5g0x",
}


def main():
    POSES_PATH.mkdir(parents=True, exist_ok=True)

    for dataset_name, file_id in DATASETS.items():
        dataset_path = POSES_PATH / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        zip_path = POSES_PATH / f"{dataset_name}.zip"

        print(f"Downloading {dataset_name} from file ID: {file_id}")
        gdown.download(id=file_id, output=str(zip_path), quiet=False)

        if not zip_path.exists():
            raise Exception(
                f"Download failed for dataset '{dataset_name}'. Ensure the Google Drive link is set to "
                "'Anyone with the link can view'."
            )

        print(f"Extracting {dataset_name}...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(dataset_path)

        zip_path.unlink()


if __name__ == "__main__":
    print("Downloading datasets from Google Drive...")
    main()
    print("Finished downloading datasets from Google Drive!")