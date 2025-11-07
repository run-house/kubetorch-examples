import requests
from tqdm import tqdm
import os
from google.cloud import storage


def download_file(url, output_path):
    print(f"Downloading {output_path.name}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return output_path


def gcs_upload(local_path, gcs_bucket, gcs_prefix, creds_path):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print('HIT')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)

    def upload_file(file_info):
        file_path, blob_name = file_info
        bucket.blob(blob_name).upload_from_filename(str(file_path))
        return blob_name

    local_path = Path(local_path)
    files_to_upload = [
        (fp, f"{gcs_prefix}/{fp.relative_to(local_path)}")
        for fp in local_path.rglob('*') if fp.is_file()
    ]

    print(f"Uploading {len(files_to_upload)} files to gs://{gcs_bucket}/{gcs_prefix}/...")
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(upload_file, f) for f in files_to_upload]
        for i, future in enumerate(as_completed(futures), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(files_to_upload)} uploaded")

    print(f"Uploaded to gs://{gcs_bucket}/{gcs_prefix}/")
