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

# 1. If custom Ngrok domain is configured
if [ -n "$NGROK_DOMAIN" ]; then
    echo "https://$NGROK_DOMAIN"
    exit 0
fi

# 2. Check tunnel log for Cloudflare URL
if [ -f "$LOG_FILE" ]; then
    URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$LOG_FILE" | tail -n 1)
    if [ -n "$URL" ]; then
        echo "$URL"
        exit 0
    fi
fi

# 3. Fallback to journalctl
JOURNAL_URL=$(journalctl --user -u satquery-tunnel.service -n 50 --no-pager 2>/dev/null | grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' | tail -n 1)
if [ -n "$JOURNAL_URL" ]; then
    echo "$JOURNAL_URL"
    exit 0
fi

echo "Tunnel connecting..."
