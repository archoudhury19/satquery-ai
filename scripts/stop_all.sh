#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Stopping SatQuery AI 24/7 Services"
echo "============================================================"

echo "[+] Stopping satquery-tunnel.service..."
systemctl --user stop satquery-tunnel.service 2>/dev/null || true

echo "[+] Stopping satquery.service..."
systemctl --user stop satquery.service 2>/dev/null || true

echo "[+] Services stopped successfully."
echo "============================================================"
