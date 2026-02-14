# DINOv3 model download folder

This folder stores DINOv3 checkpoint files downloaded by the helper script.

## Download command

Run from repository root:

```bash
./scripts/download_dinov3_models.sh
```

The script now:
- Shows a numbered list of available checkpoints.
- Prints a short use-case description for each model.
- Lets you choose one, many, or all models interactively.
- Saves files with clean checkpoint filenames (avoids very long query-string-based names).
- If a download returns HTTP 403, prompts for a fresh signed URL so you can retry only that model.

## Why HTTP 403 can happen

Many DINOv3 links are signed/temporary and may expire or require account-gated access.
If the default URL fails, paste a fresh signed URL when prompted by the script.
