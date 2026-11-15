#!/usr/bin/env bash
# One-time AWS setup for .github/workflows/deploy.yml. Run it on your own machine with AWS CLI v2
# and credentials that may create IAM roles, for example:
#   AWS_PROFILE=<your-profile> deploy/setup_aws.sh
# It creates or updates only the resources named below and is safe to re-run.
# deploy/teardown_aws.sh removes them again.
set -euo pipefail

REGION="${AWS_REGION:-eu-central-1}"
REPO="${GITHUB_REPO:-isgandat/animal-drug-safety}"
# GitHub's immutable OIDC subject for this repository (the owner and repository IDs are public).
# Check with: gh api repos/$REPO/actions/oidc/customization/sub --jq .sub_claim_prefix
SUB_PREFIX="${OIDC_SUB_PREFIX:-repo:isgandat@136094683/animal-drug-safety@1456862230}"
ECR_REPOSITORY="animal-drug-safety"
FUNCTION_NAME="animal-drug-safety-api"
DEPLOY_ROLE="ads-github-deploy"
EXEC_ROLE="ads-lambda-exec"
LOG_GROUP="/aws/lambda/$FUNCTION_NAME"
OIDC_HOST="token.actions.githubusercontent.com"
export AWS_REGION="$REGION" AWS_PAGER=""

account=$(aws sts get-caller-identity --query Account --output text)
echo "In AWS account $account, region $REGION, this will create or update:"
echo "  - ECR repository $ECR_REPOSITORY (scan on push, keeps the 3 newest images)"
echo "  - IAM roles $DEPLOY_ROLE (for GitHub Actions) and $EXEC_ROLE (for the function)"
echo "  - log group $LOG_GROUP (14-day retention)"
echo "  - the GitHub OIDC identity provider, only if the account does not have one yet"
read -r -p "Continue? [y/N] " answer
case "$answer" in y|Y) ;; *) echo "Nothing changed."; exit 1 ;; esac

if gh_prefix=$(gh api "repos/$REPO/actions/oidc/customization/sub" --jq .sub_claim_prefix 2>/dev/null) \
    && [ -n "$gh_prefix" ] && [ "$gh_prefix" != "$SUB_PREFIX" ]; then
  echo "GitHub reports the subject prefix $gh_prefix; using it."
  SUB_PREFIX=$gh_prefix
fi

repo_arn="arn:aws:ecr:${REGION}:${account}:repository/${ECR_REPOSITORY}"
function_arn="arn:aws:lambda:${REGION}:${account}:function:${FUNCTION_NAME}"
provider_arn="arn:aws:iam::${account}:oidc-provider/${OIDC_HOST}"

# --- container registry -------------------------------------------------------------------
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPOSITORY" \
    --image-scanning-configuration scanOnPush=true >/dev/null
  echo "created ECR repository $ECR_REPOSITORY"
fi
aws ecr put-lifecycle-policy --repository-name "$ECR_REPOSITORY" --lifecycle-policy-text '{
  "rules": [{"rulePriority": 1, "description": "keep the 3 newest images",
             "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 3},
             "action": {"type": "expire"}}]}' >/dev/null

# --- GitHub OIDC provider: short-lived credentials for the workflow, no stored access keys ---
if clients=$(aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$provider_arn" \
    --query ClientIDList --output text 2>/dev/null); then
  echo "reusing the existing GitHub OIDC provider"
  if ! grep -qwF sts.amazonaws.com <<<"$clients"; then
    aws iam add-client-id-to-open-id-connect-provider --open-id-connect-provider-arn "$provider_arn" \
      --client-id sts.amazonaws.com
    echo "added the sts.amazonaws.com audience to it"
  fi
else
  aws iam create-open-id-connect-provider --url "https://$OIDC_HOST" --client-id-list sts.amazonaws.com >/dev/null
  echo "created the GitHub OIDC provider"
fi

ensure_role() {  # name, trust policy
  if aws iam get-role --role-name "$1" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$1" --policy-document "$2"
  else
    aws iam create-role --role-name "$1" --assume-role-policy-document "$2" >/dev/null
    echo "created IAM role $1"
  fi
}

# --- role for the function: write its own logs, pull its own image ---------------------------
ensure_role "$EXEC_ROLE" '{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                 "Action": "sts:AssumeRole"}]}'
aws iam put-role-policy --role-name "$EXEC_ROLE" --policy-name run-function --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"logs:CreateLogStream\", \"logs:PutLogEvents\"],
     \"Resource\": \"arn:aws:logs:${REGION}:${account}:log-group:${LOG_GROUP}:*\"},
    {\"Effect\": \"Allow\", \"Action\": [\"ecr:BatchGetImage\", \"ecr:GetDownloadUrlForLayer\"],
     \"Resource\": \"${repo_arn}\"}]}"

# --- role for GitHub Actions: only main and v* tags of this repository may assume it ----------
ensure_role "$DEPLOY_ROLE" "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Effect\": \"Allow\",
    \"Principal\": {\"Federated\": \"${provider_arn}\"},
    \"Action\": \"sts:AssumeRoleWithWebIdentity\",
    \"Condition\": {
      \"StringEquals\": {\"${OIDC_HOST}:aud\": \"sts.amazonaws.com\"},
      \"StringLike\": {\"${OIDC_HOST}:sub\": [\"${SUB_PREFIX}:ref:refs/heads/main\",
                                             \"${SUB_PREFIX}:ref:refs/tags/v*\"]}}}]}"
aws iam put-role-policy --role-name "$DEPLOY_ROLE" --policy-name deploy-animal-drug-safety --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Sid\": \"EcrLogin\", \"Effect\": \"Allow\", \"Action\": \"ecr:GetAuthorizationToken\", \"Resource\": \"*\"},
    {\"Sid\": \"PushImage\", \"Effect\": \"Allow\",
     \"Action\": [\"ecr:BatchCheckLayerAvailability\", \"ecr:InitiateLayerUpload\", \"ecr:UploadLayerPart\",
                \"ecr:CompleteLayerUpload\", \"ecr:PutImage\", \"ecr:BatchGetImage\",
                \"ecr:GetDownloadUrlForLayer\", \"ecr:GetRepositoryPolicy\", \"ecr:SetRepositoryPolicy\"],
     \"Resource\": \"${repo_arn}\"},
    {\"Sid\": \"ManageFunction\", \"Effect\": \"Allow\",
     \"Action\": [\"lambda:GetFunction\", \"lambda:CreateFunction\", \"lambda:UpdateFunctionCode\",
                \"lambda:PutFunctionConcurrency\", \"lambda:GetFunctionUrlConfig\",
                \"lambda:CreateFunctionUrlConfig\", \"lambda:AddPermission\"],
     \"Resource\": \"${function_arn}\"},
    {\"Sid\": \"PassExecRole\", \"Effect\": \"Allow\", \"Action\": \"iam:PassRole\",
     \"Resource\": \"arn:aws:iam::${account}:role/${EXEC_ROLE}\",
     \"Condition\": {\"StringEquals\": {\"iam:PassedToService\": \"lambda.amazonaws.com\"}}}]}"

# --- logs ------------------------------------------------------------------------------------
if [ -z "$(aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" \
    --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName" --output text)" ]; then
  aws logs create-log-group --log-group-name "$LOG_GROUP"
fi
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14

# --- hand the role to GitHub as a secret (keeps the account ID out of the public logs) -------
role_arn="arn:aws:iam::${account}:role/${DEPLOY_ROLE}"
if command -v gh >/dev/null && gh secret set AWS_DEPLOY_ROLE_ARN --repo "$REPO" --body "$role_arn"; then
  echo "saved the role ARN as the GitHub secret AWS_DEPLOY_ROLE_ARN"
else
  echo "Now add this repository secret on GitHub (Settings > Secrets and variables > Actions):"
  echo "  AWS_DEPLOY_ROLE_ARN = $role_arn"
fi
echo "Done. Run the Deploy workflow on GitHub (Actions > Deploy > Run workflow)."
