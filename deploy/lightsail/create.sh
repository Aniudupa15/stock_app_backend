#!/usr/bin/env bash
# Create the F&O bot server: AWS Lightsail Ubuntu in Mumbai + a static IP (free while attached).
# Run from the repo root in Git Bash on your PC (uses your configured aws CLI):
#   bash deploy/lightsail/create.sh
# Shows the monthly price and asks before creating anything. Idempotent: re-running reuses
# what already exists. Prints the static IP to whitelist in Kite (Profile -> IP Whitelist).
set -euo pipefail
REGION="${AWS_REGION:-ap-south-1}"
NAME="${NAME:-fno-bot}"
BUNDLE="${BUNDLE:-nano_3_1}"        # smallest plan with a public IPv4 (512 MB RAM; setup adds swap)
BLUEPRINT="${BLUEPRINT:-ubuntu_24_04}"
ZONE="${ZONE:-${REGION}a}"
KEY="${KEY:-$HOME/.ssh/$NAME.pem}"
lsail() { command aws --region "$REGION" lightsail "$@" | tr -d '\r'; }

# --- sanity: the plan and image must exist in this region
PRICE=$(lsail get-bundles --query "bundles[?bundleId=='$BUNDLE'].price | [0]" --output text)
if [ "$PRICE" = "None" ] || [ -z "$PRICE" ]; then
  echo "Bundle $BUNDLE not offered in $REGION. Available Linux bundles:"
  lsail get-bundles --query "bundles[?supportedPlatforms[0]=='LINUX_UNIX'].[bundleId,price,ramSizeInGb]" --output table
  exit 1
fi
if [ "$(lsail get-blueprints --query "blueprints[?blueprintId=='$BLUEPRINT'] | length(@)" --output text)" = "0" ]; then
  echo "Image $BLUEPRINT not offered. Ubuntu images:"
  lsail get-blueprints --query "blueprints[?contains(blueprintId,'ubuntu')].blueprintId" --output text
  exit 1
fi

if ! lsail get-instance --instance-name "$NAME" >/dev/null 2>&1; then
  echo "About to create Lightsail instance '$NAME' ($BUNDLE, $BLUEPRINT) in $ZONE at \$$PRICE/month"
  echo "(new Lightsail accounts may get the first months of this plan free - check the console)."
  read -r -p "Type YES to create it: " ok
  [ "$ok" = "YES" ] || { echo "Cancelled - nothing created."; exit 1; }

  if [ ! -f "$KEY" ]; then
    mkdir -p "$(dirname "$KEY")"
    lsail create-key-pair --key-pair-name "$NAME" --query privateKeyBase64 --output text > "$KEY"
    chmod 600 "$KEY"
    echo "SSH private key saved to $KEY (keep it private; never share or commit it)"
  fi
  lsail create-instances --instance-names "$NAME" --availability-zone "$ZONE" \
    --blueprint-id "$BLUEPRINT" --bundle-id "$BUNDLE" --key-pair-name "$NAME" >/dev/null
fi

echo -n "Waiting for the instance to run"
for _ in $(seq 60); do
  [ "$(lsail get-instance-state --instance-name "$NAME" --query state.name --output text)" = "running" ] && break
  echo -n "."; sleep 5
done; echo

if ! lsail get-static-ip --static-ip-name "$NAME-ip" >/dev/null 2>&1; then
  lsail allocate-static-ip --static-ip-name "$NAME-ip" >/dev/null
fi
if [ "$(lsail get-static-ip --static-ip-name "$NAME-ip" --query staticIp.isAttached --output text)" != "True" ]; then
  lsail attach-static-ip --static-ip-name "$NAME-ip" --instance-name "$NAME" >/dev/null
fi
IP=$(lsail get-static-ip --static-ip-name "$NAME-ip" --query staticIp.ipAddress --output text)

echo
echo "Server ready. Static IP: $IP"
echo "Next:"
echo "  1. Whitelist $IP in Kite (Profile -> IP Whitelist) and set KITE_ALLOWED_IPS=$IP in .env"
echo "  2. bash deploy/lightsail/push.sh $IP $KEY"
