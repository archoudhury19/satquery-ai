#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/stop_all.sh"
sleep 1
"$SCRIPT_DIR/start_all.sh"
