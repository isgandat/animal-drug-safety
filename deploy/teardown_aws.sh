#!/usr/bin/env bash
# Remove everything deploy/setup_aws.sh and the deploy workflow created.
#   AWS_PROFILE=<your-profile> deploy/teardown_aws.sh
# The GitHub OIDC provider is left in place because other projects may use it.
set -euo pipefail

REGION="${AWS_REGION:-eu-central-1}"
REPO="${GITHUB_REPO:-isgandat/animal-drug-safety}"
ECR_REPOSITORY="animal-drug-safety"
FUNCTION_NAME="animal-drug-safety-api"
LOG_GROUP="/aws/lambda/$FUNCTION_NAME"
export AWS_REGION="$REGION" AWS_PAGER=""

account=$(aws sts get-caller-identity --query Account --output text)
echo "In AWS account $account, region $REGION, this will DELETE the function $FUNCTION_NAME and its URL,"
echo "the ECR repository $ECR_REPOSITORY with all images, the log group $LOG_GROUP"
echo "and the IAM roles ads-github-deploy and ads-lambda-exec."
read -r -p "Type 'delete' to continue: " answer
[ "$answer" = "delete" ] || { echo "Nothing changed."; exit 1; }

if aws lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws lambda delete-function --function-name "$FUNCTION_NAME"  # also removes its URL and permissions
  echo "deleted function $FUNCTION_NAME"
fi
if aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1; then
  aws ecr delete-repository --repository-name "$ECR_REPOSITORY" --force >/dev/null
  echo "deleted ECR repository $ECR_REPOSITORY"
fi
if [ -n "$(aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" \
    --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName" --output text)" ]; then
  aws logs delete-log-group --log-group-name "$LOG_GROUP"
  echo "deleted log group $LOG_GROUP"
fi
for role in ads-github-deploy ads-lambda-exec; do
  aws iam get-role --role-name "$role" >/dev/null 2>&1 || continue
  for policy in $(aws iam list-role-policies --role-name "$role" --query PolicyNames --output text); do
    aws iam delete-role-policy --role-name "$role" --policy-name "$policy"
  done
  for arn in $(aws iam list-attached-role-policies --role-name "$role" \
      --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name "$role" --policy-arn "$arn"
  done
  aws iam delete-role --role-name "$role"
  echo "deleted IAM role $role"
done
if command -v gh >/dev/null; then
  gh secret delete AWS_DEPLOY_ROLE_ARN --repo "$REPO" 2>/dev/null && echo "deleted the GitHub secret" || true
fi
echo "Done."
