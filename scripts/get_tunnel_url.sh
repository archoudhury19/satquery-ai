#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
LOG_FILE="$ROOT_DIR/logs/tunnel.log"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# 1. Check tunnel log for Cloudflare URL (Primary)
if [ -f "$LOG_FILE" ]; then
    URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$LOG_FILE" | tail -n 1)
    if [ -n "$URL" ]; then
        echo "$URL"
        exit 0
    fi
fi

# 2. Check journalctl for Cloudflare URL
JOURNAL_URL=$(journalctl --user -u satquery-tunnel.service -n 50 --no-pager 2>/dev/null | grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' | tail -n 1)
if [ -n "$JOURNAL_URL" ]; then
    echo "$JOURNAL_URL"
    exit 0
fi

# 3. If Tailscale Funnel is active, fallback to *.ts.net URL
TS_BIN="/home/arc/.local/bin/tailscale"
TS_SOCK="/home/arc/.local/share/tailscale/tailscaled.sock"
if [ -x "$TS_BIN" ] && [ -S "$TS_SOCK" ]; then
    TS_URL=$("$TS_BIN" --socket="$TS_SOCK" funnel status 2>/dev/null | grep -o 'https://[a-zA-Z0-9.-]*\.ts\.net' | head -n 1)
    if [ -n "$TS_URL" ]; then
        echo "$TS_URL"
        exit 0
    fi
fi

echo "Tunnel connecting..."
