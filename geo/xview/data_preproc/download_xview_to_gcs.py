from pathlib import Path
import geo.xview.data_preproc.tile_xview_dataset as tile_xview_dataset
from geo.xview.data_preproc.utils import download_file, gcs_upload
import kubetorch as kt

XVIEW_URLS = {
    "train_images": {
        "url": "https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1762402412&Signature=eNEA-7E~luHR7lTovCdhFRnGj3ir1E2uL8Gm9UDe8WPca-CnlISlaBCXwDCHVGnScGjUcz~Oj4HTVDW2wZf~z8wfGqPSn-7blGfQaoUgFAaVlkcazsozZ4Rx40f1bmcjGrcVpCIYBNgqmdN2HuOAD3Am6syL~ucgbITLsfUcRsyZ2kxjTrGRnwK1ldh7M06czsXQpnaghJbC176AM76-1JS~9Rvwya1t877S-dXDKEHE2kT0C5esp9efbHG3OsGQIvzgTnnSbMMElL1b9lc95oeas0TWDOqAAcjtfT3IngqZqpYIO0yaPUFBRITSdmMAzpMqkK3ZoVglEaAjiPlJYg__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ",
    },
    "train_labels": {
        "url": "https://d307kc0mrhucc3.cloudfront.net/train_labels.tgz?Expires=1762402412&Signature=DSAbmxuZ5yw9E9r4lgCsSwQiQqY9M4BsX37T7Jhun6aD7X9iXAyU3PmVXFwj3xVLZyAKJ-Dq6riOXkqmXHI1rWEOJ-MfSHFBZdcekBQtPbAxzwj8O4iOGBnRmSWMlnFL~nsABMZFi1iwxRiTQxgcYUdSnIoYtGfe3K4bJ3z3iq8ki-yfls7S9JfXfoml-1zqLJQJscQDRlyRSRzTK5j9F7Ij5kbbEGk03WXgIEHeIBrMPNly0v5An9vdAPg-pYohTfKcmxUEplMmQ~KMz-43aRRjHzuIjxwnwbEV4Cw9-iHie4L2~~PgB~jXsk6jQtmkIP-mraOoWUXZuHNFsHO~1Q__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ",
    },
    "val_images": {
        "url": "https://d307kc0mrhucc3.cloudfront.net/val_images.tgz?Expires=1762402412&Signature=joP2Cwvs8GvQ~FsycYvUrmqcNIWzf7Enyiq1MgVAFqG~aG~8seQmjM2cjLv9YAFOoy1DJn5NTBGJYw3~o0yQbE5elABdCEJ84sR43g0SV8dYzeOvFfGGQXSvxyJ0E25lmvGUYw7TVDPFtOjAiGSban7s1n-0ofUT~WZhRtmhVYZlX2uFjP8j3NSsY0valBMOiG67KbOm4spADKEPp9rgHFznyg562oPitGFzpI-q~tfoUgHlPpLZQK8a9ovH84nW9RCZRrQwZ3M8eLdPjiNlWxF2YGCgWgc7fIIAWkKMgElmViI8WtpwpjQBzNmumLioWZEDvofVe~2aZRah2f~qdw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ",
    },
}


def download_and_upload_xview(
    download_dir, gcs_bucket, gcs_prefix, skip_raw_upload, skip_processed_upload, creds_path
):
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract each split
    for split_name, info in XVIEW_URLS.items():
        tgz_path = download_dir / f"{split_name}.tgz"
        extract_dir = download_dir / split_name

        if tgz_path.exists() or extract_dir.exists():
            print(f"Found existing {tgz_path.name}")
        else:
            download_file(info["url"], tgz_path)

        if not extract_dir.exists():
            print(f"Extracting {tgz_path.name}...")
            subprocess.run(
                ["tar", "-xzf", str(tgz_path), "-C", str(download_dir)],
                check=True
            )
        else:
            print(f"{split_name} already extracted")

    if not skip_raw_upload:
        print("Uploading to Google Cloud Storage")
        for split_name in XVIEW_URLS.keys():
            split_dir = download_dir / split_name
            if split_dir.exists():
                gcs_upload(split_dir, gcs_bucket, f"{gcs_prefix}/{split_name}", creds_path)
    else:
        print(f"All data downloaded to {download_dir.absolute()}")

    tile_xview_dataset(
        images_dir=f"{download_dir}/train_images",
        geojson_file=f"{download_dir}/xView_train.geojson",
        output_dir=f"{download_dir}_tiled/train",
        tile_size=768,
        overlap=0.33,
        max_workers=4
    )

    if not skip_processed_upload:
        gcs_upload(f"{download_dir}_tiled/train", gcs_bucket, f"{gcs_prefix}_tiled/train", creds_path)


if __name__ == "__main__":

    download_dir = "./xview_data"
    gcs_bucket = "rh_demo_image_data"
    gcs_prefix = "xview"
    skip_raw_upload = True
    skip_processed_upload = False
    local_creds_path = "/Users/paulyang/Downloads/runhouse-test-8ad14c2b4edf.json"

    compute = kt.Compute(cpus=4,
                         disk_size="100Gi",
                         image=kt.Image().pip_install(["tqdm", "google-cloud-storage", "numpy", "Pillow"]),
                         secrets=[kt.Secret(name="gcp-dataaccess", path=local_creds_path,)]
                         )

    remote_download = kt.fn(download_and_upload_xview).to(compute)
    remote_download(download_dir, gcs_bucket, gcs_prefix, skip_raw_upload, skip_processed_upload, local_creds_path)
