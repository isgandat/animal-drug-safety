#!/usr/bin/env bash
# Create or update the Lambda function from an image in ECR, then print its public URL.
# Used by .github/workflows/deploy.yml after deploy/setup_aws.sh has been run once.
#   deploy/deploy_lambda.sh <image-uri>
set -euo pipefail

IMAGE="${1:?usage: deploy_lambda.sh <image-uri>}"
FUNCTION_NAME="${FUNCTION_NAME:-animal-drug-safety-api}"
EXEC_ROLE_NAME="${EXEC_ROLE_NAME:-ads-lambda-exec}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
export AWS_PAGER=""

# Ignore "already exists" so a half-finished first run can simply be re-run.
add_statement() {
  local sid=$1 out
  shift
  if ! out=$(aws lambda add-permission --function-name "$FUNCTION_NAME" --statement-id "$sid" \
      --principal '*' "$@" 2>&1 >/dev/null); then
    grep -q ResourceConflictException <<<"$out" || { echo "$out" >&2; return 1; }
  fi
}

set +e
lookup=$(aws lambda get-function --function-name "$FUNCTION_NAME" 2>&1 >/dev/null)
found=$?
set -e

if [ "$found" -eq 0 ]; then
  echo "Updating $FUNCTION_NAME"
  aws lambda update-function-code --function-name "$FUNCTION_NAME" --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME"
elif grep -q ResourceNotFoundException <<<"$lookup"; then
  echo "Creating $FUNCTION_NAME (first deployment)"
  account=$(aws sts get-caller-identity --query Account --output text)
  aws lambda create-function --function-name "$FUNCTION_NAME" \
    --package-type Image --code "ImageUri=$IMAGE" --architectures x86_64 \
    --role "arn:aws:iam::${account}:role/${EXEC_ROLE_NAME}" \
    --memory-size 1024 --timeout 30 \
    --description "Triage API of github.com/isgandat/animal-drug-safety" >/dev/null
  aws lambda wait function-active-v2 --function-name "$FUNCTION_NAME"
  # Cap parallel runs so a flood of requests to the public URL cannot run up costs.
  aws lambda put-function-concurrency --function-name "$FUNCTION_NAME" \
    --reserved-concurrent-executions "$MAX_CONCURRENCY" >/dev/null \
    || echo "::warning::could not reserve concurrency (the account limit may be too low); the API still works"
else
  echo "$lookup" >&2
  exit 1
fi

# Public HTTPS endpoint. Since October 2025 it needs both permission statements, added separately.
if ! aws lambda get-function-url-config --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FUNCTION_NAME" --auth-type NONE >/dev/null
fi
add_statement public-url --action lambda:InvokeFunctionUrl --function-url-auth-type NONE
add_statement public-url-invoke --action lambda:InvokeFunction --invoked-via-function-url

url=$(aws lambda get-function-url-config --function-name "$FUNCTION_NAME" --query FunctionUrl --output text)
echo "Function URL: $url"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "FUNCTION_URL=$url" >> "$GITHUB_ENV"
fi
