#!/usr/bin/env bash
# Backup of the MLOps platform: the database (pg_dump) and the dataset volume (pages; the augmented pages are
# rebuilt by the platform).
#   scripts/backup.sh /media/<encrypted-disk>/madoc-mlops-backup
# Run from the repository folder (docker compose finds the project).
#
# Every run keeps a dated copy: the last 14 database dumps and the last 14 copies of the pages. A copy of the pages
# takes almost no space: the files that did not change are hard links to the previous copy (rsync --link-dest),
# so a page deleted or overwritten by mistake can still be found in an older copy.
#
# Restore: the database with
#   docker compose exec -T db pg_restore -U mlops -d mlops --clean < <backup>/db/mlops-<date>.dump
# and the pages of the same date with
#   docker run --rm -v madoc-mlops_data:/data -v <backup>/data/<date>:/backup:ro alpine:3.20 \
#     sh -c 'apk add --no-cache rsync >/dev/null && rsync -a --delete /backup/pages/ /data/pages/'
set -euo pipefail
target="${1:?usage: scripts/backup.sh <backup folder on an encrypted disk>}"
keep="${MLOPS_BACKUP_KEEP:-14}"
mkdir -p "$target/db" "$target/data"
stamp=$(date +%Y%m%d-%H%M)
docker compose exec -T db pg_dump -U mlops -d mlops --format=custom > "$target/db/mlops-$stamp.dump.part"
mv "$target/db/mlops-$stamp.dump.part" "$target/db/mlops-$stamp.dump"   # a dump cut half-way never looks complete
ls -1t "$target"/db/mlops-*.dump | tail -n +"$((keep + 1))" | xargs -r rm --
# the pages and their XML (the dataset), in data/<date>/pages, linked to the previous copy
docker run --rm -e STAMP="$stamp" -e KEEP="$keep" -v madoc-mlops_data:/data:ro -v "$target/data":/backup \
  alpine:3.20 sh -c '
    set -eu
    apk add --no-cache rsync >/dev/null
    rm -rf /backup/*.part                                                      # a copy cut half-way
    previous=$(ls -1d /backup/2*/ 2>/dev/null | sort | tail -n 1)
    [ -z "$previous" ] && [ -d /backup/pages ] && previous=/backup/            # the copy of the old layout
    rsync -a ${previous:+--link-dest="${previous%/}/pages"} /data/pages/ "/backup/$STAMP.part/pages/"
    mv "/backup/$STAMP.part" "/backup/$STAMP"
    ls -1d /backup/2*/ | sort | awk -v keep="$KEEP" "{ line[NR] = \$0 } END { for (i = 1; i <= NR - keep; i++) print line[i] }" \
      | xargs -r rm -rf
  '
echo "backup done: $target (database and pages of $stamp)"