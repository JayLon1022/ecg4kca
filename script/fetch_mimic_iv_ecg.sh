#!/usr/bin/env bash
set -euo pipefail

# Parse command line arguments
SHOW_PROGRESS=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --show-progress)
      SHOW_PROGRESS=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--show-progress]"
      exit 1
      ;;
  esac
done

BASE_URL="https://physionet.org/files/mimic-iv-ecg/1.0/files"
DEST_DIR="/Volumes/Black/Data/ecg4kca/mimic-iv-ecg/1.0/files"

mkdir -p "$DEST_DIR"

for i in {6..7}; do
  P="p100${i}"
  echo "→ Downloading $P"
  
  if [ "$SHOW_PROGRESS" = true ]; then
    wget --quiet --show-progress \
         -r -N -c -np -nH --cut-dirs=4 \
         -P "$DEST_DIR" \
         "$BASE_URL/$P/"
  else
    wget --quiet \
         -r -N -c -np -nH --cut-dirs=4 \
         -P "$DEST_DIR" \
         "$BASE_URL/$P/"
  fi
done

echo "✓ All done!"