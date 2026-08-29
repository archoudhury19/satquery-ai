#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

TITLE="${1:-SatQuery AI Notification}"
MESSAGE="${2:-Server status update}"
URL=$("$SCRIPT_DIR/get_tunnel_url.sh" 2>/dev/null || echo "Unknown URL")

PAYLOAD="🛰️ **$TITLE**\n$MESSAGE\n🌐 **Public URL**: $URL\n⚡ **GPU**: NVIDIA GeForce MX450 (CUDA)"

# 1. Send Discord Webhook if configured
if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    curl -s -H "Content-Type: application/json" \
         -X POST \
         -d "{\"content\": \"$PAYLOAD\"}" \
         "$DISCORD_WEBHOOK_URL" > /dev/null 2>&1 || true
fi

# 2. Send Telegram Message if configured
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    TEXT=$(echo -e "$PAYLOAD")
    curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
         -d "chat_id=$TELEGRAM_CHAT_ID" \
         -d "text=$TEXT" > /dev/null 2>&1 || true
fi
