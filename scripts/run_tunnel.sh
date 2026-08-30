#!/usr/bin/env bash
set -e

ROOT_DIR="/home/arc/.gemini/antigravity/scratch/satquery-ai"
ENV_FILE="$ROOT_DIR/.env"
mkdir -p "$ROOT_DIR/logs"

# Auto-truncate log if it exceeds 5MB for 24/7 disk preservation
if [ -f "$ROOT_DIR/logs/tunnel.log" ] && [ "$(stat -c%s "$ROOT_DIR/logs/tunnel.log" 2>/dev/null || echo 0)" -gt 5242880 ]; then
    tail -n 2000 "$ROOT_DIR/logs/tunnel.log" > "$ROOT_DIR/logs/tunnel.log.tmp"
    mv "$ROOT_DIR/logs/tunnel.log.tmp" "$ROOT_DIR/logs/tunnel.log"
fi

# Load environment variables if available
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "[+] Starting Cloudflare Zero Trust Permanent Tunnel..."
    "$ROOT_DIR/scripts/notify.sh" "SatQuery AI Online" "Permanent Cloudflare Zero Trust tunnel is active!" >/dev/null 2>&1 || true
    exec /home/arc/.local/bin/cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN"
elif [ -n "$NGROK_AUTHTOKEN" ] && [ -n "$NGROK_DOMAIN" ]; then
    echo "[+] Starting Ngrok Free Custom Domain Tunnel ($NGROK_DOMAIN)..."
    /home/arc/.local/bin/ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
    exec /home/arc/.local/bin/ngrok http --domain="$NGROK_DOMAIN" 8000 --log=stdout --log-level=info
else
    # Background watcher: send Telegram alert with new URL for quick tunnel
    (
        sleep 4
        for i in {1..30}; do
            URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$ROOT_DIR/logs/tunnel.log" 2>/dev/null | tail -n 1)
            if [ -n "$URL" ]; then
                # Wait 3s for Cloudflare edge DNS to propagate
                sleep 3
                "$ROOT_DIR/scripts/notify.sh" "SatQuery AI Online" "Live 24/7 server ready." >/dev/null 2>&1 || true
                break
            fi
            sleep 1
        done
    ) &
    echo "[+] Starting Cloudflare Quick Tunnel with HTTP2 fallback..."
    exec /home/arc/.local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --protocol http2 --logfile "$ROOT_DIR/logs/tunnel.log" --no-autoupdate
fi
