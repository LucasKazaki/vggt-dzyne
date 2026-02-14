#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/models/dinov3"

REPO_DIR="$ROOT_DIR"

mkdir -p "$TARGET_DIR"

MODELS=(
  "dinov3_vits16_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth|Small ViT-S/16 backbone. Fastest option for feature extraction and quick experiments."
  "dinov3_vits16plus_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_vits16plus/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth|ViT-S/16+ variant. Good balance between speed and stronger representations than ViT-S."
  "dinov3_vitb16_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth|ViT-B/16 backbone. Strong default for transfer learning with moderate GPU memory."
  "dinov3_vitl16_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth|ViT-L/16 backbone. Higher quality features for dense tasks if you have more compute."
  "dinov3_vith16plus_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth|Large ViT-H/16+ model. Best quality among standard checkpoints, very compute intensive."
  "dinov3_vit7b16_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth|7B-parameter giant model for highest accuracy research settings."
  "dinov3_vitl16_pretrain_sat493m|https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth|Satellite-focused ViT-L checkpoint, useful for remote sensing imagery."
  "dinov3_convnext_small_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_convnext_small/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth|ConvNeXt-S backbone. Good for CNN-oriented pipelines and lower memory budgets."
  "dinov3_convnext_base_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_convnext_base/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth|ConvNeXt-B backbone. Solid all-round CNN checkpoint for downstream tasks."
  "dinov3_convnext_large_pretrain_lvd1689m|https://dinov3.llamameta.net/dinov3_convnext_large/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth|ConvNeXt-L backbone. Higher-capacity CNN features for quality-critical workloads."
  "dinov3_vit7b16_imagenet1k_linear_head|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth|ImageNet-1K linear classification head for ViT-7B evaluation."
  "dinov3_vit7b16_coco_detr_head|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_coco_detr_head-b0235ff7.pth|COCO DETR detection head for object detection benchmarking."
  "dinov3_vit7b16_ade20k_m2f_head|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth|ADE20K Mask2Former segmentation head for semantic segmentation."
  "dinov3_vit7b16_synthmix_dpt_head|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth|Depth prediction (DPT) head trained on SynthMix."
  "dinov3_vitl16_dinotxt_vision_head_and_text_encoder|https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth|Vision-text checkpoint for cross-modal retrieval and zero-shot text/image matching."
)

print_menu() {
  echo "Available DINOv3 models:"
  local idx=1
  for model in "${MODELS[@]}"; do
    IFS='|' read -r name _url desc <<<"$model"
    printf "  %2d) %-52s %s\n" "$idx" "$name" "$desc"
    ((idx++))
  done
  echo
  echo "Enter model numbers separated by commas (e.g. 1,3), or 'all'."
}

download_one() {
  local url="$1"
  local out_path="$2"

  echo "Downloading: $(basename "$out_path")"

  if wget -c --tries=3 --timeout=30 -O "$out_path" "$url"; then
    return 0
  fi

  echo "⚠️  Download failed for $(basename "$out_path")."
  echo "    If you got HTTP 403, the host may require fresh signed links from Meta's release page or account access."
  echo "    You can retry with a signed URL for this specific file when prompted."

  read -r -p "Paste a signed URL to retry (or press Enter to skip): " signed_url
  if [[ -z "$signed_url" ]]; then
    echo "Skipped $(basename "$out_path")"
    return 1
  fi

  wget -c --tries=3 --timeout=30 -O "$out_path" "$signed_url"
}


upload_with_release() {
  local files=("$@")
  local default_repo
  local repo
  local tag
  local title
  local archive_name
  local archive_path

  if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI (gh) is not installed. Skipping release upload."
    return 1
  fi

  default_repo="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"
  read -r -p "GitHub repo for release upload [${default_repo:-owner/repo}]: " repo
  repo="${repo:-$default_repo}"
  if [[ -z "$repo" ]]; then
    echo "Repository is required for release upload."
    return 1
  fi

  tag="dinov3-models-$(date +%Y%m%d-%H%M%S)"
  read -r -p "Release tag [$tag]: " input_tag
  tag="${input_tag:-$tag}"

  title="DINOv3 model bundle $tag"
  read -r -p "Release title [$title]: " input_title
  title="${input_title:-$title}"

  archive_name="$tag.tar.gz"
  archive_path="$TARGET_DIR/$archive_name"
  tar -czf "$archive_path" -C "$TARGET_DIR" "${files[@]##*/}"

  if gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
    gh release upload "$tag" "$archive_path" --repo "$repo" --clobber
  else
    gh release create "$tag" "$archive_path" --repo "$repo" --title "$title" --notes "DINOv3 selected model bundle uploaded by script."
  fi

  echo "Release upload complete: https://github.com/$repo/releases/tag/$tag"
}

upload_with_git() {
  local files=("$@")
  local branch
  local commit_message

  if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Current directory is not a git repository. Skipping git upload."
    return 1
  fi

  echo "Preparing git upload from local machine."
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "⚠️  git-lfs is not installed. Large model files may exceed GitHub limits."
    echo "    Install git-lfs and run 'git lfs install' for reliable model uploads."
  fi

  branch="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
  read -r -p "Branch to push models [$branch]: " input_branch
  branch="${input_branch:-$branch}"

  if [[ "$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)" != "$branch" ]]; then
    git -C "$REPO_DIR" checkout "$branch"
  fi

  git -C "$REPO_DIR" add -- "${files[@]}"
  commit_message="Add selected DINOv3 model checkpoints"
  read -r -p "Commit message [$commit_message]: " input_message
  commit_message="${input_message:-$commit_message}"
  git -C "$REPO_DIR" commit -m "$commit_message"
  git -C "$REPO_DIR" push origin "$branch"

  echo "Git upload complete to branch '$branch'."
}

print_menu
echo
echo "Optional: if you have a signed URL query token from Meta (starts with 'Policy='),"
echo "paste it once and it will be appended to all selected model URLs."
read -r -p "Shared signed query token (optional): " shared_query
print_menu
read -r -p "Selection: " selection

if [[ -z "${selection// }" ]]; then
  echo "No selection provided. Exiting."
  exit 1
fi

selected_indices=()
if [[ "$selection" =~ ^[Aa][Ll][Ll]$ ]]; then
  for ((i = 1; i <= ${#MODELS[@]}; i++)); do
    selected_indices+=("$i")
  done
else
  selection="${selection// /}"
  IFS=',' read -r -a raw_indices <<<"$selection"
  for idx in "${raw_indices[@]}"; do
    if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
      echo "Invalid selection '$idx'. Use numbers separated by commas, or 'all'." >&2
      exit 1
    fi
    if ((idx < 1 || idx > ${#MODELS[@]})); then
      echo "Selection '$idx' is out of range 1-${#MODELS[@]}." >&2
      exit 1
    fi
    selected_indices+=("$idx")
  done
fi

printf "\nDownloading %d selected model(s) into %s\n\n" "${#selected_indices[@]}" "$TARGET_DIR"

failed=0
for idx in "${selected_indices[@]}"; do
  entry="${MODELS[$((idx - 1))]}"
  IFS='|' read -r _name url _desc <<<"$entry"

  filename="$(basename "${url%%\?*}")"
  output_path="$TARGET_DIR/$filename"

  if ! download_one "$url" "$output_path"; then
    failed=1
  fi
  echo
 done

if ((failed)); then
  echo "Finished with some skipped/failed downloads."
  exit 1
fi

echo "Done."
