#!/usr/bin/env bash
# Create/update the WhatsApp alert Lambda (S3 session bucket, IAM role, function, public URL guarded by
# a secret header, concurrency 1, optional $1 budget alert). Idempotent - re-run to deploy code changes.
#   bash deploy/whatsapp_lambda/deploy.sh [alert-email-for-$1-budget]
# Needs: aws CLI configured (`aws configure`), node + npm, git (baileys pulls libsignal from GitHub).
set -euo pipefail
ALERT_EMAIL="${1:-}"
REGION="${AWS_REGION:-ap-south-1}"
FN=whatsapp-notify
ROLE=whatsapp-notify-role
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

aws() { command aws --region "$REGION" "$@" | tr -d '\r'; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="whatsapp-notify-$ACCOUNT"
echo "Account $ACCOUNT, region $REGION, bucket $BUCKET"

# --- S3 bucket for the session: private, encrypted (SSE-S3 default), versioned so a bad write can be rolled back.
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-old-sessions","Status":"Enabled","Filter":{},"NoncurrentVersionExpiration":{"NoncurrentDays":7}}]}'

# --- IAM role: CloudWatch logs + read/write of the one session object.
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting for the new IAM role to propagate..."; sleep 15
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name session-object --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":[\"arn:aws:s3:::$BUCKET/session.json\",\"arn:aws:s3:::$BUCKET/session.lock\"]},{\"Effect\":\"Allow\",\"Action\":\"s3:ListBucket\",\"Resource\":\"arn:aws:s3:::$BUCKET\",\"Condition\":{\"StringEquals\":{\"s3:prefix\":[\"session.json\",\"session.lock\"]}}}]}"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)

# --- Build: production deps only, no peer deps (sharp/jimp are optional image extras with native binaries).
rm -rf build function.zip && mkdir build
cp index.mjs session-blob.mjs package.json build/
(cd build && npm install --omit=dev --legacy-peer-deps --no-audit --no-fund --loglevel=error)
# Git Bash's GNU tar can't write zips (it would silently produce a tarball), so use Windows' bsdtar or zip.
ZIP_FILES=(index.mjs session-blob.mjs package.json node_modules)
if [ -x /c/Windows/System32/tar.exe ]; then
  (cd build && /c/Windows/System32/tar.exe -a -cf ../function.zip "${ZIP_FILES[@]}")
else
  (cd build && zip -qr ../function.zip "${ZIP_FILES[@]}")
fi
unzip -l function.zip >/dev/null 2>&1 || node -e "const b=require('fs').readFileSync('function.zip');if(b.readUInt32LE(0)!==0x04034b50)throw new Error('function.zip is not a zip')"

# --- Secret for the x-api-key header: generated once, kept in .deploy.env (gitignored).
[ -f .deploy.env ] && source .deploy.env
API_SECRET="${WHATSAPP_NOTIFY_SECRET:-$(node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")}"
ENV_VARS="Variables={SESSION_BUCKET=$BUCKET,SESSION_KEY=session.json,API_SECRET=$API_SECRET}"

# --- Function (no VPC, so no NAT gateway cost).
if aws lambda get-function --function-name "$FN" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FN" --zip-file fileb://function.zip >/dev/null
  aws lambda wait function-updated --function-name "$FN"
  aws lambda update-function-configuration --function-name "$FN" --environment "$ENV_VARS" \
    --timeout 90 --memory-size 1024 >/dev/null
else
  aws lambda create-function --function-name "$FN" --runtime nodejs22.x --architectures x86_64 \
    --handler index.handler --role "$ROLE_ARN" --zip-file fileb://function.zip \
    --timeout 90 --memory-size 1024 --environment "$ENV_VARS" >/dev/null
fi
aws lambda wait function-updated --function-name "$FN"

# Also cap concurrency at 1 where the account allows it. New accounts (total limit 10) can't reserve
# any - that's fine, the handler's S3 lock (session.lock) already serialises sends.
aws lambda put-function-concurrency --function-name "$FN" --reserved-concurrent-executions 1 >/dev/null \
  2>/dev/null || echo "Note: account too new to reserve concurrency; relying on the S3 lock instead."

# --- Public Function URL; the handler rejects anything without the secret header.
if ! aws lambda get-function-url-config --function-name "$FN" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FN" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url --action lambda:InvokeFunctionUrl \
    --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url-invoke --action lambda:InvokeFunction \
    --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
fi
URL=$(aws lambda get-function-url-config --function-name "$FN" --query FunctionUrl --output text)

# --- Optional $1/month budget alert (AWS Budgets: first two budgets are free).
if [ -n "$ALERT_EMAIL" ] && ! aws budgets describe-budget --account-id "$ACCOUNT" --budget-name one-dollar-alert >/dev/null 2>&1; then
  aws budgets create-budget --account-id "$ACCOUNT" \
    --budget '{"BudgetName":"one-dollar-alert","BudgetLimit":{"Amount":"1","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
    --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":1,\"ThresholdType\":\"ABSOLUTE_VALUE\"},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$ALERT_EMAIL\"}]}]"
  echo "Budget alert created for $ALERT_EMAIL"
fi

umask 077
printf 'WHATSAPP_NOTIFY_URL=%s\nWHATSAPP_NOTIFY_SECRET=%s\nWHATSAPP_SESSION_BUCKET=%s\n' "$URL" "$API_SECRET" "$BUCKET" > .deploy.env
rm -rf build function.zip
echo
echo "Deployed. URL: $URL"
echo "Values saved to deploy/whatsapp_lambda/.deploy.env - copy WHATSAPP_NOTIFY_URL / WHATSAPP_NOTIFY_SECRET into the repo .env"
