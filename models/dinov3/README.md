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
- Accepts an optional shared signed query token (for temporary Meta links) and appends it to all selected model URLs.
- Saves files with clean checkpoint filenames (avoids very long query-string-based names).
- If a download returns HTTP 403, prompts for a fresh signed URL so you can retry only that model.
- After successful downloads, offers optional upload to GitHub:
  - `g`: commit/push selected checkpoints to the repo branch (GitHub file page).
  - `r`: create/upload a `.tar.gz` bundle to GitHub Releases using `gh` CLI.

## Why HTTP 403 can happen

Many DINOv3 links are signed/temporary and may expire or require account-gated access.
If the default URL fails, paste a fresh signed URL when prompted by the script.

## Running on your local machine

Use this script on your own machine (where you have access credentials and no restrictive proxy):

1. Clone this repository.
2. Run `./scripts/download_dinov3_models.sh`.
3. Pick the model(s) to download.
4. Choose upload mode:
   - `g` to push files directly to your GitHub repository branch.
   - `r` to upload one archive to GitHub Releases.
