#!/usr/bin/env bash
# Backup of the MLOps platform: the database (pg_dump) and the dataset volume (pages, augmented pages are rebuilt).
#   scripts/backup.sh /media/<encrypted-disk>/madoc-mlops-backup
# Run from the repository folder (docker compose finds the project). Keeps the last 14 database dumps.
set -euo pipefail
target="${1:?usage: scripts/backup.sh <backup folder on an encrypted disk>}"
mkdir -p "$target/db" "$target/data"
stamp=$(date +%Y%m%d-%H%M)
docker compose exec -T db pg_dump -U mlops -d mlops --format=custom > "$target/db/mlops-$stamp.dump"
ls -1t "$target"/db/mlops-*.dump | tail -n +15 | xargs -r rm --
# the pages and their XML (the dataset); augmented/, work/ and uploads/ can be rebuilt or are temporary
docker run --rm -v madoc-mlops_data:/data:ro -v "$target/data":/backup alpine:3.20 \
  sh -c 'apk add --no-cache rsync >/dev/null && rsync -a --delete /data/pages/ /backup/pages/'
echo "backup done: $target (database $stamp, pages)"
