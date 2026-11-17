#!/usr/bin/env bash
# Create the F&O bot server: AWS Lightsail Ubuntu in Mumbai + a static IP (free while attached).
# Run from the repo root in Git Bash on your PC (uses your configured aws CLI):
#   bash deploy/lightsail/create.sh
# Shows the monthly price and asks before creating anything. Idempotent: re-running reuses
# what already exists. Prints the static IP to whitelist in Kite (Profile -> IP Whitelist).
set -euo pipefail
# Retry transient network errors on reads (creates are guarded by the list checks below).
export AWS_RETRY_MODE="${AWS_RETRY_MODE:-standard}" AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-6}"
REGION="${AWS_REGION:-ap-south-1}"
NAME="${NAME:-fno-bot-server}"   # instance (and "$NAME-ip" static IP)
KEYNAME="${KEYNAME:-fno-bot}"     # SSH key pair - kept apart from the instance name (Lightsail
                                  # rejected an instance named like the existing key pair)
BUNDLE="${BUNDLE:-nano_3_1}"        # smallest plan with a public IPv4 (512 MB RAM; setup adds swap)
BLUEPRINT="${BLUEPRINT:-ubuntu_24_04}"
ZONE="${ZONE:-${REGION}a}"
KEY="${KEY:-$HOME/.ssh/$KEYNAME.pem}"
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

# Existence checks LIST resources instead of treating any error as "not found": on a flaky
# connection a failed lookup must stop the script, not trigger a duplicate create.
INSTANCES=$(lsail get-instances --query "length(instances[?name=='$NAME'])" --output text)
if [ "$INSTANCES" = "0" ]; then
  echo "About to create Lightsail instance '$NAME' ($BUNDLE, $BLUEPRINT) in $ZONE at \$$PRICE/month"
  echo "(new Lightsail accounts may get the first months of this plan free - check the console)."
  read -r -p "Type YES to create it: " ok
  [ "$ok" = "YES" ] || { echo "Cancelled - nothing created."; exit 1; }

  KEYPAIRS=$(lsail get-key-pairs --query "length(keyPairs[?name=='$KEYNAME'])" --output text)
  if [ "$KEYPAIRS" != "0" ] && [ ! -f "$KEY" ]; then
    echo "Key pair '$KEYNAME' exists in AWS but $KEY is missing - delete the key pair in the"
    echo "Lightsail console (Account -> SSH keys) or set KEY=<path to the saved .pem>, then re-run."
    exit 1
  fi
  if [ "$KEYPAIRS" = "0" ]; then
    mkdir -p "$(dirname "$KEY")"
    lsail create-key-pair --key-pair-name "$KEYNAME" --query privateKeyBase64 --output text > "$KEY"
    chmod 600 "$KEY"
    echo "SSH private key saved to $KEY (keep it private; never share or commit it)"
  fi
  echo "Existing Lightsail instances in $REGION: $(lsail get-instances --query 'instances[].name' --output text)"
  lsail create-instances --instance-names "$NAME" --availability-zone "$ZONE" \
    --blueprint-id "$BLUEPRINT" --bundle-id "$BUNDLE" --key-pair-name "$KEYNAME" >/dev/null
fi

echo -n "Waiting for the instance to run"
for _ in $(seq 60); do
  [ "$(lsail get-instance-state --instance-name "$NAME" --query state.name --output text)" = "running" ] && break
  echo -n "."; sleep 5
done; echo

STATIC_IPS=$(lsail get-static-ips --query "length(staticIps[?name=='$NAME-ip'])" --output text)
if [ "$STATIC_IPS" = "0" ]; then
  lsail allocate-static-ip --static-ip-name "$NAME-ip" >/dev/null
fi
ATTACHED=$(lsail get-static-ip --static-ip-name "$NAME-ip" --query staticIp.isAttached --output text)
if [ "$ATTACHED" != "True" ]; then
  lsail attach-static-ip --static-ip-name "$NAME-ip" --instance-name "$NAME" >/dev/null
fi
IP=$(lsail get-static-ip --static-ip-name "$NAME-ip" --query staticIp.ipAddress --output text)

echo
echo "Server ready. Static IP: $IP"
echo "Next:"
echo "  1. Whitelist $IP in Kite (Profile -> IP Whitelist) and set KITE_ALLOWED_IPS=$IP in .env"
echo "  2. bash deploy/lightsail/push.sh $IP $KEY"
