# Description: This script downloads the imagenet dataset from HuggingFace, preprocesses it, and uploads it to S3.
# You don't need to reproduce this if you want to just use your own dataset directly.

import kubetorch as kt


def download_preproc_and_upload(
    dataset_name,
    save_bucket_name,
    train_sample="100%",
    test_sample="100%",
    val_sample="100%",
    cache_dir="~/rh_download",
    save_s3_folder_prefix="",
):
    from datasets import DatasetDict, load_dataset
    from torchvision import transforms

    print("Downloading Data")
    # Make sure to enable access to the HuggingFace dataset first:
    # https://huggingface.co/datasets/ILSVRC/imagenet-1k
    dataset = load_dataset(
        dataset_name,
        token=True,
        trust_remote_code=True,
        split=[
            f"train[:{train_sample}]",
            f"validation[:{val_sample}]",
            f"test[:{test_sample}]",
        ],
        download_mode="reuse_cache_if_exists",
        cache_dir=f"{cache_dir}/huggingface_cache/",
    )
    print("Download complete")

    ds = DatasetDict(
        {
            "train": dataset[0],
            "validation": dataset[1],
            "test": dataset[2],
        }
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def preprocess_example(batch):
        processed_images = []
        for img in batch["image"]:
            # Check if the image is grayscale
            if img.mode != "RGB":
                img = img.convert("RGB")  # Convert grayscale to RGB
            processed_images.append(preprocess(img))
        batch["image"] = processed_images
        return batch

    print("Preprocessing Data")

    for split_name in ds.keys():
        print(split_name)
        dataset = ds[split_name].map(preprocess_example, batched=True)
        dataset.save_to_disk(f"{cache_dir}/{split_name}")
        s3_path = f"s3://{save_bucket_name}/{save_s3_folder_prefix}/{split_name}"
        dataset.save_to_disk(s3_path)

    print("Uploaded Preprocessed Data")


# ## Launch the compute
# Define the image and the compute that the work will be run on.
# The image is package installations and secrets here layered on top of a basic Ubuntu image,
# but can also take a Docker image as a base.
if __name__ == "__main__":

    img = (
        kt.images.Debian()
        .pip_install(
            [
                "torchvision",
                "Pillow",
                "datasets",
                "boto3",
                "s3fs>=2024.10.0",
            ]
        )
        .sync_secrets(["aws", "huggingface"])
    )

    compute = kt.Compute(
        cpus="32",
        disk_size="1000G",
        image=img,
    ).up_if_not()

    # Mount the disk to download the data to
    s3_bucket = "rh-demo-external"
    cache_dir = "/mnt/nvme"
    compute.run_bash(
        [
            f"sudo mkdir {cache_dir}",
            "sudo mkfs.ext4 /dev/nvme1n1",
            f"sudo mount /dev/nvme1n1 {cache_dir}",
            f"sudo chown ubuntu:ubuntu {cache_dir}",
            f"mkdir -p {cache_dir}/huggingface_cache",
        ]
    )
    # Send our download / preprocessing function to the compute we just created
    remote_preproc = kt.fn(download_preproc_and_upload).to(compute)
    # Download the data, sampling down to 15% for our example
    remote_preproc(
        dataset_name="imagenet-1k",
        save_bucket_name=s3_bucket,
        cache_dir=cache_dir,
        train_sample="15%",
        validation_sample="25%",
        test_sample="15%",
        save_s3_folder_prefix="resnet-training-example/preprocessed_imagenet",
    )
