#!/usr/bin/env bash
# Run a LOCAL LiveKit SFU on this machine and reach its SIGNALING endpoint from
# the Modal policy server over YOUR TAILNET — the Tailscale sibling of
# `local_sfu.sh` (which uses a Cloudflare quick tunnel instead). Same SFU, same
# generated config, same `livekit.local.env`; the ONLY difference is how Modal
# reaches signaling.
#
#     robot_*.py ──ws://127.0.0.1:7880──► local livekit-server ◄──ws:// over tailnet── modal (signaling)
#                                                   ▲
#                                                   └────────── UDP :7882 ◄──────────── modal (media+data)
#
# Why bother: the quick tunnel hands out a NEW random trycloudflare.com hostname
# every launch (so every Modal run needs fresh flags) and publishes your SFU's
# signaling endpoint to the whole internet, unauthenticated. The tailnet URL is
# stable and only your devices can reach it.
#
# What the tailnet does and does NOT carry (the load-bearing fact — unchanged
# from local_sfu.sh):
#   * SIGNALING (join, offer/answer, trickle ICE) rides the tailnet: the Modal
#     container joins as an ephemeral userspace node and dials
#     ws://<this-mac's 100.x>:7880. That hop is WireGuard-encrypted end to end,
#     so plain `ws://` is correct here — no TLS, no cert, no tunnel.
#   * WebRTC media AND data channels (the 30fps observation/action loop) are
#     STILL SRTP/SCTP over ICE — UDP straight from the Modal container to this
#     machine's PUBLIC IP, NOT through Tailscale. `use_external_ip: true` makes
#     the SFU STUN-discover that IP and advertise `public_ip:7882` (single-port
#     UDP mux); ICE hole-punches the NAT from both ends. Tailscale changes
#     nothing about this path — if your venue's NAT (CGNAT / symmetric) defeats
#     hole punching, media still fails; forward UDP 7882 to this machine.
#
# First run generates ~/.cache/huggingface/lerobot/livekit.local.yaml (random API
# key/secret) and every run writes ~/.cache/huggingface/lerobot/livekit.local.env so
# the ROBOT side automatically targets the local SFU (makermodslab.drtc._env.load_env
# loads it with override=True). Delete livekit.local.env to point the robot back at
# LiveKit Cloud. NOTE: it outlives this script — after Ctrl-C the robot keeps dialing
# ws://127.0.0.1:7880 and gets connection refused until you restart this script or
# delete the file.
#
# Requires: livekit-server (`brew install livekit`) and Tailscale, installed and
# logged in on this Mac (https://tailscale.com/download/mac — the App Store
# build puts the CLI at /Applications/Tailscale.app/Contents/MacOS/Tailscale).
# Modal side additionally needs a `tailscale-auth` secret; see
# docs/drtc/README.md "Modal secrets".
set -euo pipefail
# All generated state lives beside the rest of MakerMods Lab's persistent state
# (see makermodslab/utils/config.py: DRTC_SFU_CONFIG_PATH / DRTC_LOCAL_ENV_PATH /
# DRTC_LOG_DIR), NOT next to this script, so the robot side can be started from
# any directory — makermodslab.drtc._env.load_env reads the override from there.
STATE_DIR="${HOME}/.cache/huggingface/lerobot"
LOG_DIR="${STATE_DIR}/logs/drtc"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

CONFIG="${STATE_DIR}/livekit.local.yaml"
ROBOT_ENV="${STATE_DIR}/livekit.local.env"
SIGNAL_PORT=7880
TCP_PORT=7881   # ICE/TCP fallback (only reachable on the LAN; harmless to keep)
UDP_PORT=7882   # single-port UDP mux — the ONE port to forward if NAT blocks media
TS=$(date +%s)
mkdir -p "$LOG_DIR"

# --- this Mac's tailnet address ----------------------------------------------
# Resolved BEFORE anything is started: a missing/logged-out Tailscale is the one
# failure mode worth reporting up front rather than after the SFU is running.
TAILSCALE_BIN="$(command -v tailscale || true)"
if [[ -z "$TAILSCALE_BIN" && -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
    TAILSCALE_BIN=/Applications/Tailscale.app/Contents/MacOS/Tailscale
fi
if [[ -z "$TAILSCALE_BIN" ]]; then
    echo "[local_sfu_ts] tailscale CLI not found."
    echo "               Install Tailscale (https://tailscale.com/download/mac or"
    echo "               'brew install tailscale' + 'sudo tailscaled install-system-daemon'),"
    echo "               log in, then re-run. For the Cloudflare-tunnel variant instead:"
    echo "               ${REPO_ROOT}/tools/drtc/local_sfu.sh"
    exit 1
fi
TS_IP="$("$TAILSCALE_BIN" ip -4 2>/dev/null | head -1 || true)"
if [[ -z "$TS_IP" ]]; then
    echo "[local_sfu_ts] tailscale is installed but has no IPv4 address — not logged in"
    echo "               or not running. Bring it up (Tailscale menu-bar app, or"
    echo "               'tailscale up'), confirm '$TAILSCALE_BIN status' looks healthy,"
    echo "               and re-run."
    exit 1
fi

# --- one-time local credentials (identical to local_sfu.sh) ------------------
if [[ ! -f "$CONFIG" ]]; then
    KEY="local-$(openssl rand -hex 4)"
    SECRET="$(openssl rand -hex 32)"   # livekit requires >= 32 chars
    cat > "$CONFIG" <<EOF
# Generated by tools/drtc/local_sfu_ts.sh ($(date -u +%Y-%m-%dT%H:%M:%SZ)).
# Delete this file to rotate the local API key/secret.
port: ${SIGNAL_PORT}
rtc:
  udp_port: ${UDP_PORT}
  tcp_port: ${TCP_PORT}
  use_external_ip: true
keys:
  ${KEY}: ${SECRET}
EOF
    echo "[local_sfu_ts] generated $CONFIG (new local key/secret)"
fi
KEY=$(awk '/^keys:/{getline; gsub(/[: ]+$/,"",$1); print $1; exit}' "$CONFIG" | tr -d ':')
SECRET=$(awk '/^keys:/{getline; print $2; exit}' "$CONFIG")

# --- point the robot side at the local SFU -----------------------------------
cat > "$ROBOT_ENV" <<EOF
# Generated by tools/drtc/local_sfu_ts.sh — routes the ROBOT side to the local SFU.
# Loaded with override=True by makermodslab.drtc._env.load_env; LIVEKIT_ROOM
# still comes from livekit.env. DELETE this file to go back to LiveKit Cloud.
LIVEKIT_URL=ws://127.0.0.1:${SIGNAL_PORT}
LIVEKIT_API_KEY=${KEY}
LIVEKIT_API_SECRET=${SECRET}
EOF

# --- start SFU (no cloudflared: the tailnet is the signaling path) ------------
# livekit-server binds 0.0.0.0 by default, so the tailnet interface is already
# covered — nothing extra to configure for Modal to reach :7880 over WireGuard.
SFU_LOG="${LOG_DIR}"/local_sfu_ts_${TS}.log

livekit-server --config "$CONFIG" >"$SFU_LOG" 2>&1 &
SFU_PID=$!
trap 'echo; echo "[local_sfu_ts] shutting down"; kill $SFU_PID 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

# SFU up?
for _ in $(seq 1 20); do
    curl -sf "http://127.0.0.1:${SIGNAL_PORT}" >/dev/null 2>&1 && break
    kill -0 $SFU_PID 2>/dev/null || { echo "[local_sfu_ts] livekit-server died — see $SFU_LOG"; exit 1; }
    sleep 0.5
done

echo
echo "[local_sfu_ts] SFU:     ws://127.0.0.1:${SIGNAL_PORT}  (pid $SFU_PID, log $SFU_LOG)"
echo "[local_sfu_ts] tailnet: ws://${TS_IP}:${SIGNAL_PORT}   (signaling only, WireGuard-encrypted)"
echo "[local_sfu_ts] robot side: python -m makermodslab.drtc.robot_sync (or robot_rtc) from anywhere — $ROBOT_ENV now targets the local SFU"
echo
echo "[local_sfu_ts] Modal side (this URL is STABLE across launches — unlike the quick tunnel):"
echo
echo "    modal run ${REPO_ROOT}/makermodslab/drtc/modal_policy_rtc.py --policy-path \${HF_USER}/my_policy --horizon 16 \\"
echo "        --tailscale --livekit-url ws://${TS_IP}:${SIGNAL_PORT} \\"
echo "        --livekit-api-key ${KEY} --livekit-api-secret ${SECRET}"
echo
echo "[local_sfu_ts] (needs a one-time 'modal secret create tailscale-auth TS_AUTHKEY=tskey-...'"
echo "                with a REUSABLE + EPHEMERAL key — see docs/drtc/README.md 'Modal secrets'.)"
echo "[local_sfu_ts] Media is still direct UDP ${UDP_PORT} hole punch; Tailscale does not carry it."
echo "[local_sfu_ts] Ctrl-C to stop."
wait
