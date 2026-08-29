#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$ROOT_DIR/logs"

echo "============================================================"
echo "Starting SatQuery AI 24/7 Services (systemd)"
echo "============================================================"

# Ensure user lingering is requested for 24/7 background execution
if command -v loginctl &>/dev/null; then
    loginctl enable-linger "$USER" 2>/dev/null || true
fi

# Reload systemd user daemon
systemctl --user daemon-reload

# Start SatQuery server
echo "[+] Starting satquery.service..."
systemctl --user enable satquery.service
systemctl --user restart satquery.service

# Start Cloudflare Tunnel
echo "[+] Starting satquery-tunnel.service..."
systemctl --user enable satquery-tunnel.service
systemctl --user restart satquery-tunnel.service

# Wait for local health check
echo -n "[+] Waiting for SatQuery AI local backend to be healthy"
HEALTHY=false
for i in {1..30}; do
    if curl -s -f "http://127.0.0.1:8000/api/health" &>/dev/null; then
        HEALTHY=true
        echo " [OK]"
        break
    fi
    echo -n "."
    sleep 1
done

if [ "$HEALTHY" = false ]; then
    echo " [FAILED]"
    echo "Server failed to respond within 30s. Checking logs:"
    journalctl --user -u satquery.service -n 20 --no-pager
    exit 1
fi

# Wait for Cloudflare Tunnel URL
echo -n "[+] Discovering Cloudflare public tunnel URL"
TUNNEL_URL=""
for i in {1..20}; do
    TUNNEL_URL=$("$SCRIPT_DIR/get_tunnel_url.sh" 2>/dev/null || true)
    if [[ "$TUNNEL_URL" =~ ^https://.*\.trycloudflare\.com$ ]]; then
        echo " [OK]"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "============================================================"
echo "SatQuery AI is RUNNING 24/7!"
echo "============================================================"
echo "Local Endpoint   : http://127.0.0.1:8000"
if [ -n "$TUNNEL_URL" ]; then
    echo "Public HTTPS URL : $TUNNEL_URL"
else
    echo "Public HTTPS URL : Connecting... (run './satqueryctl url' in a moment)"
fi
echo "GPU Device       : NVIDIA GeForce MX450 (CUDA Active)"
echo "============================================================"

# Send notification if webhook/bot configured
"$SCRIPT_DIR/notify.sh" "SatQuery AI Server Online" "Server is healthy and ready for satellite inference." >/dev/null 2>&1 || true

