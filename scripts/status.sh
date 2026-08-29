#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "SatQuery AI 24/7 Server Status Dashboard"
echo "============================================================"

# Systemd Units
echo "[1] Systemd Services Status:"
echo -n "  - satquery.service        : "
systemctl --user is-active satquery.service 2>/dev/null || echo "inactive"
echo -n "  - satquery-tunnel.service : "
systemctl --user is-active satquery-tunnel.service 2>/dev/null || echo "inactive"

# Local Health
echo ""
echo "[2] Local Backend Health (http://127.0.0.1:8000/api/health):"
HEALTH_JSON=$(curl -s "http://127.0.0.1:8000/api/health" 2>/dev/null || echo "")
if [ -n "$HEALTH_JSON" ]; then
    echo "  $HEALTH_JSON"
else
    echo "  [DOWN] Local backend is not responding."
fi

# Public Tunnel URL
echo ""
echo "[3] Cloudflare Public Tunnel URL:"
TUNNEL_URL=$("$SCRIPT_DIR/get_tunnel_url.sh" 2>/dev/null || echo "Tunnel disconnected / not established")
echo "  $TUNNEL_URL"

# NVIDIA GPU Status
echo ""
echo "[4] NVIDIA GPU Status:"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        IFS=',' read -r name mem_used mem_total util temp <<< "$GPU_INFO"
        echo "  - GPU Name    : $name"
        echo "  - VRAM Used   : ${mem_used// /} MB / ${mem_total// /} MB"
        echo "  - GPU Load    : ${util// /}%"
        echo "  - Temperature : ${temp// /}°C"
    else
        echo "  - GPU info unavailable via nvidia-smi"
    fi
fi

echo "============================================================"
