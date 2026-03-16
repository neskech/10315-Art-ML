import re
import requests
import os
from bs4 import BeautifulSoup
import tqdm

# Set up logging

# From https://docsaid.org/en/blog/download-from-google-drive-using-python/


def download_from_google(file_id: str, file_name: str, target: str = "."):
    """
    Downloads a file from Google Drive, handling potential confirmation tokens for large files.

    Args:
        file_id (str):
            The ID of the file to download from Google Drive.
        file_name (str):
            The name to save the downloaded file as.
        target (str, optional):
            The directory to save the file in. Defaults to the current directory (".").

    Raises:
        Exception: If the download fails or the file cannot be created.

    Notes:
        This function handles both small and large files. For large files, it automatically processes
        Google's confirmation token to bypass warnings about virus scans or file size limits.

    Example:
        Download a file to the current directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt"
            )

        Download a file to a specific directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt",
                target="./downloads"
            )
    """
    # First try: docs.google.com/uc?export=download&id=FileID
    base_url = "https://docs.google.com/uc"
    session = requests.Session()
    params = {
        "export": "download",
        "id": file_id
    }
    response = session.get(base_url, params=params, stream=True)

    # If Content-Disposition is present, the file is directly available
    if "content-disposition" not in response.headers:
        # Try to get the token from cookies
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        # If no token in cookies, extract it from the HTML
        if not token:
            soup = BeautifulSoup(response.text, "html.parser")
            # Common case: HTML contains a form with id="download-form"
            download_form = soup.find("form", {"id": "download-form"})
            if download_form and download_form.get("action"):
                # Extract action URL, which might be drive.usercontent.google.com/download
                download_url = download_form["action"]
                # Collect all hidden inputs
                hidden_inputs = download_form.find_all(
                    "input", {"type": "hidden"})
                form_params = {}
                for inp in hidden_inputs:
                    if inp.get("name") and inp.get("value") is not None:
                        form_params[inp["name"]] = inp["value"]

                # Re-send the GET request with these parameters
                response = session.get(
                    download_url, params=form_params, stream=True) # type: ignore
            else:
                # Otherwise, search for confirm=xxx in HTML
                match = re.search(r'confirm=([0-9A-Za-z-_]+)', response.text)
                if match:
                    token = match.group(1)
                    # Include the confirm token in the request
                    params["confirm"] = token
                    response = session.get(
                        base_url, params=params, stream=True)
                else:
                    raise Exception(
                        "Unable to find the download link or confirmation token in the response. Download failed.")

        else:
            # Use the token obtained from cookies and resend the request
            params["confirm"] = token
            response = session.get(base_url, params=params, stream=True)

    # Ensure the download directory exists
    os.makedirs(target, exist_ok=True)
    file_path = os.path.join(target, file_name)

    # Start downloading the file in chunks, with a progress bar
    try:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, "wb") as f, tqdm.tqdm(
            desc=file_name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"File successfully downloaded to: {file_path}")

    except Exception as e:
        raise Exception(f"File download failed: {e}")
