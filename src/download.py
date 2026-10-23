"""Download the UCI 'Default of Credit Card Clients' dataset (CC BY 4.0)."""
import io
import urllib.request
import zipfile

from config import DATA_URL, RAW_DIR, RAW_XLS


def download() -> None:
    if RAW_XLS.exists():
        print(f"Already present: {RAW_XLS.name}")
        return
    with urllib.request.urlopen(DATA_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract(RAW_XLS.name, RAW_DIR)
    print(f"Downloaded {RAW_XLS.name}")


if __name__ == "__main__":
    download()
