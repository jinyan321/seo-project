#!/bin/sh
# Daily pg_dump into /backups (mounted to ./backups on the host), keeping $KEEP_DAYS days.
# Copy ./backups off the server (e.g. rclone to S3/B2 from a host cron) so a lost disk
# doesn't lose the backups too.
set -eu
while true; do
  f="/backups/tracker-$(date -u +%Y%m%dT%H%M%SZ).sql.gz"
  if pg_dump --no-owner | gzip > "$f.tmp"; then
    mv "$f.tmp" "$f"
    echo "backup ok: $f"
  else
    rm -f "$f.tmp"
    echo "backup FAILED" >&2
  fi
  find /backups -name 'tracker-*.sql.gz' -mtime +"$KEEP_DAYS" -delete
  sleep 86400
done
