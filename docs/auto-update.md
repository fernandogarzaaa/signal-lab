# Auto-update for the demo box

The live demo box (`signal-lab-demo`, port 3100) is a **pure deploy
target**: it serves whatever is checked out on its disk, not what is on
GitHub `main`. This script closes that gap. Once installed, every merge
to `main` reaches the live URL within ~5 minutes, with no SSH session
needed.

## How it works

`scripts/auto-update.sh` runs from cron every 5 minutes and:

1. `git fetch origin main` (a failed fetch is a silent no-op; next tick retries)
2. Compares `HEAD` to `origin/main`; exits silently when already current
3. On change: `git reset --hard origin/main`
4. Reinstalls node deps (`packages/cli`, `--omit=dev`) only when a
   package manifest changed in the update
5. `sudo systemctl restart <service>`
6. Health-checks `http://127.0.0.1:3100/` for up to 30s; logs loudly on failure

Single-flight via `flock`, so overlapping ticks never stack.

## One-time install (AWS console, ~2 minutes)

SSH is not required. Open the EC2 console, select instance
`i-0ead85f4c02139ea3` (`signal-lab-demo`), and press **Connect**.

Find the repo checkout and the service name:

```bash
ps aux | grep '[n]ode.*server.js' | awk '{print $NF}' | xargs -I{} dirname {} | xargs -I{} dirname {}
systemctl list-units --type=service --state=running | grep -i signal
```

Then install the cron (replace the paths with what the commands above printed):

```bash
(crontab -l 2>/dev/null; echo "*/5 * * * * SIGNAL_LAB_REPO=/opt/signal-lab SIGNAL_LAB_SERVICE=signal-lab /opt/signal-lab/scripts/auto-update.sh >> /var/log/signal-lab-update.log 2>&1") | crontab -
```

Notes:

- The cron user needs passwordless `sudo systemctl restart <service>`
  (the default `ubuntu` user has this).
- The log file needs to be writable by the cron user; if
  `/var/log/signal-lab-update.log` is not, point the redirect at
  `~/signal-lab-update.log` instead.
- Verify with `tail -f` on the log, then merge something small and
  watch the box pick it up.

## Safety rules

- The box keeps **no local edits**. `reset --hard` discards anything
  not committed; commit work in the repo instead.
- Only merge to `main` after green CI. A merged-then-broken commit
  auto-deploys broken, and the script's only guard is the
  post-restart health check.
- If the health check fails, the script exits 1 and logs `ERROR`;
  check the log, fix forward on `main`, and the next tick redeploys.
