from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from k8s_client import create_job, get_job_status


# THIS must be named exactly `app`
app = FastAPI(title="K8s ML Training Control Plane")


class JobRequest(BaseModel):
    job_name: str
    image: str = "trainer:latest"
    epochs: int = 2
    dataset_size: int = 0      # 0 = full MNIST
    batch_size: int = 64
    world_size: int = 2        # number of DDP processes




@app.post("/jobs")
def submit_job(req: JobRequest):
    try:
        env = {
            "EPOCHS": req.epochs,
            "DATASET_SIZE": req.dataset_size,
            "BATCH_SIZE": req.batch_size,
            "WORLD_SIZE": req.world_size,
        }
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
