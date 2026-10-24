#!/usr/bin/env bash
# Create/update the zero-cost momentum bot Lambda: private S3 state bucket, IAM roles, Python function,
# public Function URL (Kite redirect target), and two EventBridge Scheduler crons (Asia/Kolkata).
# Idempotent - re-run to deploy code/config changes.
#   bash deploy/momentum_lambda/deploy.sh
# Reads secrets from the repo-root .env (never printed). Needs: aws CLI, the repo .venv.
set -euo pipefail
REGION="${AWS_REGION:-ap-south-1}"
FN=momentum-bot
ROLE=momentum-bot-role
SCHED_ROLE=momentum-bot-scheduler-role
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
cd "$HERE"

aws() { command aws --region "$REGION" "$@" | tr -d '\r'; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="momentum-bot-$ACCOUNT"
echo "Account $ACCOUNT, region $REGION, bucket $BUCKET"

# --- Private, versioned state bucket (ledger, today's Kite session, last basket).
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-old-versions","Status":"Enabled","Filter":{},"NoncurrentVersionExpiration":{"NoncurrentDays":30}}]}'

# Seed the ledger from the PC bot once (never overwrites the cloud copy).
if ! aws s3api head-object --bucket "$BUCKET" --key state.json >/dev/null 2>&1; then
  if [ -f "$ROOT/scripts/state/zerodha_bot_state.json" ]; then
    aws s3 cp "$ROOT/scripts/state/zerodha_bot_state.json" "s3://$BUCKET/state.json" >/dev/null
    echo "Seeded state.json from the PC bot ledger"
  fi
fi

# --- Lambda role: logs + the three state objects.
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting for the new IAM role to propagate..."; sleep 15
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name state-objects --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":[\"arn:aws:s3:::$BUCKET/state.json\",\"arn:aws:s3:::$BUCKET/session.json\",\"arn:aws:s3:::$BUCKET/basket.json\"]},{\"Effect\":\"Allow\",\"Action\":\"s3:ListBucket\",\"Resource\":\"arn:aws:s3:::$BUCKET\"}]}"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)

# --- Build: handler + the SAME tested planning/safety modules + pure-Python deps.
rm -rf build function.zip && mkdir -p build/services/trading_service/momentum build/libs/trading_calendar
cp handler.py kite.py db.py build/
for d in services services/trading_service services/trading_service/momentum libs libs/trading_calendar; do : > "build/$d/__init__.py"; done
cp "$ROOT/services/trading_service/momentum/live_plan.py" "$ROOT/services/trading_service/momentum/live_safety.py" build/services/trading_service/momentum/
cp "$ROOT/libs/trading_calendar/calendar.py" build/libs/trading_calendar/
"$PY" -m pip install -q -r requirements.txt --target build --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp
if [ -x /c/Windows/System32/tar.exe ]; then
  (cd build && /c/Windows/System32/tar.exe -a -cf ../function.zip *)
else
  (cd build && zip -qr ../function.zip .)
fi

# --- Environment from the repo .env (written to a temp file, never echoed).
"$PY" - "$ROOT/.env" "$BUCKET" > env.json <<'PYEOF'
import json, sys
from dotenv import dotenv_values
v = dotenv_values(sys.argv[1])
keys = ["KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID", "BASKET_API_KEY", "BOT_CAPITAL_CAP", "BOT_TOP_N",
        "BOT_STOP_LOSS_PCT", "BOT_MIN_DAYS_BETWEEN", "NSE_HOLIDAYS", "ZERODHA_LIVE", "WHATSAPP_NOTIFY_URL",
        "WHATSAPP_NOTIFY_SECRET", "WHATSAPP_TO", "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_FROM", "BOT_MAIL_TO", "MAIL_TO"]
env = {k: v[k] for k in keys if v.get(k)}
env["DATABASE_URL"] = (v.get("MOMENTUM_DB_URL") or v["DATABASE_URL"]).replace("postgresql+asyncpg://", "postgresql://")
env["STATE_BUCKET"] = sys.argv[2]
print(json.dumps({"Variables": env}))
PYEOF
chmod 600 env.json

if aws lambda get-function --function-name "$FN" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FN" --zip-file fileb://function.zip >/dev/null
  aws lambda wait function-updated --function-name "$FN"
  aws lambda update-function-configuration --function-name "$FN" --environment file://env.json \
    --timeout 300 --memory-size 512 >/dev/null
else
  aws lambda create-function --function-name "$FN" --runtime python3.12 --architectures x86_64 \
    --handler handler.lambda_handler --role "$ROLE_ARN" --zip-file fileb://function.zip \
    --timeout 300 --memory-size 512 --environment file://env.json >/dev/null
fi
aws lambda wait function-updated --function-name "$FN"
rm -f env.json
FN_ARN=$(aws lambda get-function --function-name "$FN" --query Configuration.FunctionArn --output text)

# --- Public Function URL = Kite redirect target (/kite/callback). Tokens are only accepted after
# Kite validates them with our API secret, so the open URL can't act on its own.
if ! aws lambda get-function-url-config --function-name "$FN" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FN" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url --action lambda:InvokeFunctionUrl \
    --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url-invoke --action lambda:InvokeFunction \
    --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
fi
URL=$(aws lambda get-function-url-config --function-name "$FN" --query FunctionUrl --output text)

# --- Scheduler role + the two weekday crons (EventBridge Scheduler: 14M invocations/month free).
if ! aws iam get-role --role-name "$SCHED_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$SCHED_ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  sleep 10
fi
aws iam put-role-policy --role-name "$SCHED_ROLE" --policy-name invoke-bot --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"$FN_ARN\"}]}"
SCHED_ARN=$(aws iam get-role --role-name "$SCHED_ROLE" --query Role.Arn --output text)
for spec in "momentum-morning|cron(20 9 ? * MON-FRI *)|morning" "momentum-evening|cron(15 19 ? * MON-FRI *)|evening"; do
  IFS='|' read -r NAME EXPR JOB <<<"$spec"
  ARGS=(--name "$NAME" --schedule-expression "$EXPR" --schedule-expression-timezone Asia/Kolkata
        --flexible-time-window Mode=OFF
        --target "{\"Arn\":\"$FN_ARN\",\"RoleArn\":\"$SCHED_ARN\",\"Input\":\"{\\\"job\\\":\\\"$JOB\\\"}\",\"RetryPolicy\":{\"MaximumRetryAttempts\":2}}")
  if aws scheduler get-schedule --name "$NAME" >/dev/null 2>&1; then
    aws scheduler update-schedule "${ARGS[@]}" >/dev/null
  else
    aws scheduler create-schedule "${ARGS[@]}" >/dev/null
  fi
done

rm -rf build function.zip
echo
echo "Deployed $FN."
echo "Kite redirect URL to set on developers.kite.trade: ${URL%/}/kite/callback"
