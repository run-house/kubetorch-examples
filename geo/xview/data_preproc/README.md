# XView Data Preprocessing

Downloads XView satellite dataset, tiles into 768x768 chips with 33% overlap, and uploads to GCS.

**Usage:**
```bash
python download_xview_to_gcs.py
```

**Output:** YOLO-format tiles (images + labels) ready for training.
