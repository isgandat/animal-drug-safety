# Deploying to AWS Lambda

The triage API runs on AWS Lambda as a container image behind a public HTTPS function URL.
Only the model ships, not the 0.9 GB database, so `/triage`, `/health` and `/docs` work while
`/signals` and `/stats` answer 503.

## What the Deploy workflow does

[`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) runs on a `v*` tag or by hand from `main`:

1. lint and unit tests
2. download and load the latest openFDA export, with the 12 quality checks
3. train, calibrate and write the model card (kept as a run artifact)
4. **release gate** ([`check_metrics.py`](check_metrics.py)): stop unless test PR-AUC ≥ 0.85, the model
   beats the logistic-regression baseline, triage recall ≥ 0.85 and calibrated ECE ≤ 0.03
5. get short-lived AWS credentials through GitHub OIDC (no access keys stored in GitHub)
6. build the image ([`lambda.Dockerfile`](lambda.Dockerfile)) and push it to ECR,
   tagged `<commit>-<model version>`
7. create or update the function ([`deploy_lambda.sh`](deploy_lambda.sh))
8. smoke test the live URL ([`smoke_test.py`](smoke_test.py)): the new model version must answer `/triage`

Because the model is retrained on the newest export, its numbers can differ slightly from the
ones in the main README. `/health` reports which version is live.

## One-time setup

You need AWS CLI v2 with credentials that may create IAM roles, and optionally the GitHub CLI (`gh`)
logged in. From a clone of the repository:

```bash
AWS_PROFILE=<your-profile> deploy/setup_aws.sh
```

It shows the account and region and asks before changing anything. It creates only:

| Resource | Name | Notes |
|---|---|---|
| ECR repository | `animal-drug-safety` | scan on push, keeps the 3 newest images |
| IAM role for GitHub Actions | `ads-github-deploy` | trusted only for `main` and `v*` tags of this repository; can push to that one repository, manage that one function, and pass the role below |
| IAM role for the function | `ads-lambda-exec` | write its own logs, pull its own image |
| Log group | `/aws/lambda/animal-drug-safety-api` | 14-day retention |
| GitHub OIDC provider | `token.actions.githubusercontent.com` | only if the account has none; an existing one is reused |

It then saves the deploy role ARN as the repository secret `AWS_DEPLOY_ROLE_ARN` (or prints it so you
can add it under Settings → Secrets and variables → Actions). The ARN is a secret and the workflow masks
the account ID, so the account ID does not appear in the public Actions logs.

The trust policy uses GitHub's immutable subject, `repo:isgandat@136094683/animal-drug-safety@1456862230`
(owner and repository IDs), so a future repository that reuses the name cannot assume the role.

## Deploy

```bash
gh workflow run deploy.yml --ref main     # or Actions → Deploy → Run workflow
git tag v1.0.0 && git push origin v1.0.0  # later releases
```

The first run creates the function (1024 MB, 30 s timeout, at most 2 parallel executions) and its
public URL; later runs only update the image. The URL is in the run summary:

```bash
curl -s <function-url>health
```

## Cost and clean-up

At demo traffic this should stay within the Lambda free tier, plus ECR storage for three images
(cents per month). A budget alert in the AWS account is still a good idea. The URL is public and
unauthenticated on purpose; the concurrency cap limits how much a flood of requests can run.

Remove everything (the OIDC provider stays, because other projects may use it):

```bash
AWS_PROFILE=<your-profile> deploy/teardown_aws.sh
```

## Troubleshooting

- **Not authorized to perform sts:AssumeRoleWithWebIdentity**: the run was not on `main` or a `v*` tag,
  or the subject changed. Compare `gh api repos/isgandat/animal-drug-safety/actions/oidc/customization/sub`
  with the trust policy and re-run the setup.
- **"could not reserve concurrency" warning**: the account's concurrency limit is too low to reserve any.
  The API still works, without the cap.
- **Smoke test fails**: read the function's logs in CloudWatch (`/aws/lambda/animal-drug-safety-api`).
