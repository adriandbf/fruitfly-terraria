#!/usr/bin/env bash
# Check that the Terraria server port on the PC is reachable from this machine.
# Usage: bash server/check_connection.sh <host> [port]
set -euo pipefail
host="${1:?usage: check_connection.sh <host> [port]}"
port="${2:-7777}"
if nc -z -G 3 "$host" "$port" 2>/dev/null; then
  echo "OK: $host:$port is accepting TCP connections"
else
  echo "FAIL: cannot reach $host:$port"
  echo "Check: server is running, PC firewall rule, network profile is Private, same subnet."
  exit 1
fi
