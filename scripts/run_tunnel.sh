#!/usr/bin/env bash
set -e

ROOT_DIR="/home/arc/.gemini/antigravity/scratch/satquery-ai"
ENV_FILE="$ROOT_DIR/.env"
mkdir -p "$ROOT_DIR/logs"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "[+] Starting Cloudflare Zero Trust Permanent Tunnel..."
    "$ROOT_DIR/scripts/notify.sh" "SatQuery AI Online" "Permanent Cloudflare Zero Trust tunnel is active!" >/dev/null 2>&1 || true
    exec /home/arc/.local/bin/cloudflared tunnel --no-autoupdate --protocol http2 run --token "$CLOUDFLARE_TUNNEL_TOKEN"
elif [ -n "$NGROK_AUTHTOKEN" ] && [ -n "$NGROK_DOMAIN" ]; then
    echo "[+] Starting Ngrok Free Custom Domain Tunnel ($NGROK_DOMAIN)..."
    /home/arc/.local/bin/ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
    exec /home/arc/.local/bin/ngrok http --domain="$NGROK_DOMAIN" 8000 --log=stdout --log-level=info
fi

echo "[+] Starting Multi-Network Failover Supervisor (LAN > Wi-Fi > Mobile USB)..."

CURRENT_IFACE=$(ip route show default 2>/dev/null | awk '{print $5}' | head -n 1)

while true; do
    # Reset log and active URL before starting new instance
    rm -f "$ROOT_DIR/logs/active_url.txt"
    > "$ROOT_DIR/logs/tunnel.log"

    # Launch cloudflared with HTTP/2 (carrier compatible)
    /home/arc/.local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --protocol http2 --logfile "$ROOT_DIR/logs/tunnel.log" --no-autoupdate &
    TUNNEL_PID=$!

    # Active URL Watcher (monitors tunnel.log for new URL)
    (
        for i in {1..60}; do
            sleep 1
            URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$ROOT_DIR/logs/tunnel.log" 2>/dev/null | tail -n 1)
            if [ -n "$URL" ]; then
                echo "$URL" > "$ROOT_DIR/logs/active_url.txt"
                IFACE=$(ip route show default 2>/dev/null | awk '{print $5}' | head -n 1)
                IFACE_NAME="LAN / Wi-Fi"
                if [[ "$IFACE" =~ ^enp.*u[0-9]+ ]]; then
                    IFACE_NAME="Mobile USB Tethering"
                fi
                # Wait for Telegram API reachability before dispatching alert
                for attempt in {1..5}; do
                    if curl -s -m 4 https://api.telegram.org >/dev/null 2>&1; then
                        "$ROOT_DIR/scripts/notify.sh" "SatQuery AI ($IFACE_NAME)" "Live 24/7 server online:\n$URL"
                        break
                    fi
                    sleep 2
                done
                break
            fi
        done
    ) &

    # Network watcher loop: detect interface change or network drop
    while kill -0 "$TUNNEL_PID" 2>/dev/null; do
        sleep 5
        NEW_IFACE=$(ip route show default 2>/dev/null | awk '{print $5}' | head -n 1)
        if [ -n "$NEW_IFACE" ] && [ "$NEW_IFACE" != "$CURRENT_IFACE" ]; then
            echo "[!] Network route changed from $CURRENT_IFACE to $NEW_IFACE! Triggering auto-reconnect..."
            CURRENT_IFACE="$NEW_IFACE"
            kill "$TUNNEL_PID" 2>/dev/null || true
            sleep 2
            break
        fi
    done

    wait "$TUNNEL_PID" 2>/dev/null || true
    echo "[!] Tunnel process exited. Reconnecting in 2s..."
    sleep 2
done
