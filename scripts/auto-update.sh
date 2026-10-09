#!/usr/bin/env bash
# auto-update.sh — pull-based auto-deploy for the Signal Lab demo box.
#
# Runs from cron every few minutes. Fetches origin/main; if the box is
# behind, fast-forwards the checkout, reinstalls node deps when the
# manifest changed, restarts the systemd service, and health-checks it.
# Does nothing (silently) when already up to date.
#
# The box is a PURE DEPLOY TARGET: this script runs `git reset --hard`
# on change. Never keep local edits on the box; commit them instead.
#
# Configuration (env vars, set on the crontab line):
#   SIGNAL_LAB_REPO     absolute path of the repo checkout (required)
#   SIGNAL_LAB_SERVICE  systemd service name (default: signal-lab)
#   SIGNAL_LAB_PORT     health-check port (default: 3100)
#   SIGNAL_LAB_BRANCH   branch to track (default: main)
#
# Example crontab entry (installed once on the box):
#   */5 * * * * SIGNAL_LAB_REPO=/opt/signal-lab SIGNAL_LAB_SERVICE=signal-lab /opt/signal-lab/scripts/auto-update.sh >> /var/log/signal-lab-update.log 2>&1
set -euo pipefail

REPO="${SIGNAL_LAB_REPO:?set SIGNAL_LAB_REPO to the repo checkout path}"
SERVICE="${SIGNAL_LAB_SERVICE:-signal-lab}"
PORT="${SIGNAL_LAB_PORT:-3100}"
BRANCH="${SIGNAL_LAB_BRANCH:-main}"
LOCK="/tmp/signal-lab-auto-update.lock"

log() { echo "$(date -u +%FT%TZ) [auto-update] $*"; }

# Single-flight: skip if a previous run is still working.
exec 9>"$LOCK"
if ! flock -n 9; then
  log "another run in progress, skipping"
  exit 0
fi

cd "$REPO"

# A failed fetch (network blip) is not news; just try again next tick.
if ! git fetch origin "$BRANCH" --quiet 2>/dev/null; then
  exit 0
fi

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"
if [ "$LOCAL" = "$REMOTE" ]; then
  # No new code to deploy, but self-heal: a crashed service stays down
  # until the next commit unless we restart it here.
  if ! curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/api/health"; then
    log "service $SERVICE unhealthy on port $PORT, restarting"
    if sudo -n systemctl restart "$SERVICE" 2>/dev/null; then
      log "service $SERVICE restarted by self-heal"
    else
      log "ERROR: self-heal restart of $SERVICE failed (needs passwordless sudo)"
      exit 1
    fi
  fi
  exit 0
fi

log "updating $LOCAL -> $REMOTE"

# Pure deploy target: discard anything local, match origin exactly.
git reset --hard "origin/$BRANCH" --quiet
log "checkout now at $(git rev-parse --short HEAD)"

# Reinstall node deps only when the manifest moved.
if git diff --name-only "$LOCAL" "$REMOTE" | grep -qE '(^|/)package(-lock)?\.json$'; then
  if [ -d packages/cli ] && command -v npm >/dev/null 2>&1; then
    log "package manifest changed, running npm install"
    (cd packages/cli && npm install --omit=dev --no-audit --no-fund)
  else
    log "WARNING: manifest changed but npm/packages/cli not found; skipping install"
  fi
fi

if ! sudo -n systemctl restart "$SERVICE" 2>/dev/null; then
  log "ERROR: systemctl restart $SERVICE failed (needs passwordless sudo)"
  exit 1
fi
log "service $SERVICE restarted"

# Health check: dashboard must answer 200 on 127.0.0.1 within 30s.
for i in $(seq 1 30); do
  if curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/api/health"; then
    log "healthy on port $PORT after ${i}s"
    exit 0
  fi
  sleep 1
done

log "ERROR: service $SERVICE did not become healthy on port $PORT within 30s"
exit 1
