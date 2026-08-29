#!/usr/bin/env bash
set -e

ROOT_DIR="/home/arc/.gemini/antigravity/scratch/satquery-ai"
ENV_FILE="$ROOT_DIR/.env"
mkdir -p "$ROOT_DIR/logs"

# Load environment variables if available
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

if [ -n "$NGROK_AUTHTOKEN" ] && [ -n "$NGROK_DOMAIN" ]; then
    echo "[+] Starting Ngrok Free Custom Domain Tunnel ($NGROK_DOMAIN)..."
    /home/arc/.local/bin/ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
    echo "https://$NGROK_DOMAIN" > "$ROOT_DIR/logs/tunnel.log"
    exec /home/arc/.local/bin/ngrok http --domain="$NGROK_DOMAIN" 8000 --log=stdout --log-level=info
elif [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "[+] Starting Cloudflare Custom Named Tunnel using Token..."
    exec /home/arc/.local/bin/cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN" --logfile "$ROOT_DIR/logs/tunnel.log" --no-autoupdate
else
    echo "[+] Starting Cloudflare Quick Tunnel (http://127.0.0.1:8000)..."
    exec /home/arc/.local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --logfile "$ROOT_DIR/logs/tunnel.log" --no-autoupdate
fi
