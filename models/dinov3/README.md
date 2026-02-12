# DINOv3 model download folder

This folder is intended to contain all DINOv3 checkpoint files.

## Download command

Run from repository root:

```bash
./scripts/download_dinov3_models.sh
```

The script reads URLs from `models/dinov3/urls.txt` and downloads with `wget -c` so interrupted downloads can resume.

## Note for this environment

In this execution environment, outbound requests to `dinov3.llamameta.net` through the configured proxy fail with `Proxy tunneling failed: Forbidden`, so the files could not be downloaded here.
