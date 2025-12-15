import os
from typing import Optional

import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from k8s_client import create_job, get_job_status

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-distributed-training-datasets")   # TODO: set real bucket
S3_REGION = os.getenv("S3_REGION", "us-east-1")

s3_client = boto3.client("s3", region_name=S3_REGION)

# THIS must be named exactly `app`
app = FastAPI(title="K8s ML Training Control Plane")

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
    """
    Upload a zipped dataset (e.g. ImageFolder in a .zip) to S3.

    Returns an s3:// URI that can be passed into /jobs.
    """
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET env not configured")

    key = f"datasets/{dataset_name}/{file.filename}"

    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET, key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    s3_uri = f"s3://{S3_BUCKET}/{key}"
    return {"dataset_uri": s3_uri}


class JobRequest(BaseModel):
    job_name: str
    image: str = "trainer:latest"
    epochs: int = 2
    dataset_size: int = 0      # 0 = full MNIST
    batch_size: int = 64
    world_size: int = 2        # number of DDP processes
    dataset_type: str = "mnist"          # "mnist" or "imagefolder"
    dataset_uri: Optional[str] = None    # s3://bucket/key.zip for custom datasets





@app.post("/jobs")
def submit_job(req: JobRequest):
    try:
        env = {
            "EPOCHS": str(req.epochs),
            "DATASET_SIZE": str(req.dataset_size),
            "BATCH_SIZE": str(req.batch_size),
            "WORLD_SIZE": str(req.world_size),
            "DATASET_TYPE": req.dataset_type,
            "JOB_NAME": req.job_name,
        }

        if req.dataset_uri:
            env["DATASET_URI"] = req.dataset_uri

        # Forward AWS credentials into the training pod
        for key in [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",     # optional
            "AWS_DEFAULT_REGION",
        ]:
            value = os.getenv(key)
            if value:
                env[key] = value

        job = create_job(req.job_name, req.image, env=env)
        return {"message": "Job created", "job_name": job.metadata.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/jobs/{job_name}")
def status(job_name: str):
    try:
        s = get_job_status(job_name)
        return {
            "succeeded": s.succeeded,
            "failed": s.failed,
            "active": s.active,
            "start_time": s.start_time,
            "completion_time": s.completion_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
