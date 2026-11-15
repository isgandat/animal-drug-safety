# Image for AWS Lambda. The AWS Lambda Web Adapter runs as a Lambda extension and passes
# requests to uvicorn, so the API code is the same as in the main Dockerfile.
# The trained model is baked in: run `make train` before building.
#   docker build --platform linux/amd64 --provenance=false -f deploy/lambda.Dockerfile -t ads-lambda .
FROM python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/health \
    ADS_MODEL=/app/models/model.joblib
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY etl ./etl
COPY ml ./ml
COPY stats ./stats
COPY app ./app
COPY models/model.joblib models/metadata.json ./models/

# Lambda runs the container as a non-root user on a read-only file system; nothing here writes to disk.
# There is no database in the image, so /signals and /stats answer 503.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
