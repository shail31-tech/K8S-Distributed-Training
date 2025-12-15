# 🚀 K8S Distributed Training Platform

A Mini AWS SageMaker-Style ML Training System Using Kubernetes, FastAPI & PyTorch

## 📘 Overview

This project implements an end-to-end machine learning training platform using:

- FastAPI → Control Plane (Job Submission, dataset upload)
- Kubernetes Jobs → Orchestration & Execution
- Docker → Training Runtime Environment
- PyTorch → Model Training & Evaluation
- Minikube → Local K8s Cluster Simulation

It mirrors AWS SageMaker Training Jobs, including:

- ✔ Remote job submission
- ✔ Hyperparameter passing
- ✔ Container-based training
- ✔ Model artifact saving
- ✔ Automatic evaluation
- ✔ Distributed-training-ready code (DDP compatible)
- ✔ S3-backed dataset upload + model upload flow

---

## 🧠 Features

### 1. FastAPI Control Plane
- Endpoint: `POST /jobs` — Submits training jobs to Kubernetes
- New endpoint: `POST /datasets/upload` — Upload a zipped dataset to S3 (returns s3:// URI)
- Job hyperparameters:
  - `epochs`
  - `batch_size`
  - `dataset_size`
  - `world_size`
  - `dataset_type` (`mnist` or `butterfly_csv`)
  - `dataset_uri` (S3 URI for custom datasets)

### 2. Kubernetes Training Jobs
Each job:
- Pulls the specified trainer image (e.g. `trainer:latest`)
- Executes `train.py`
- Trains a PyTorch CNN (MNIST by default; custom CSV-based datasets supported)
- Saves the model at `/model/model.pt`
- Optionally uploads the artifact to S3 when a dataset_uri (s3://...) is provided
- Logs everything to `kubectl logs`

### 3. PyTorch Distributed Training Compatible
- Uses `torch.nn.parallel.DistributedDataParallel` (DDP)
- Uses `torch.utils.data.distributed.DistributedSampler`
- `world_size = 1` in Minikube (can scale to multi-node or GPU clusters)

### 4. Automatic Evaluation
After training completes:
```
[RANK 0] Model saved to /model/model.pt
[S3] Model upload complete.   # when applicable
=== Validation Accuracy: XX.XX% ===
```

---

## 📂 Project Structure
```
k8s-distributed-training/
│
├── control-plane/
│   ├── app.py                # FastAPI job submission + dataset upload service
│   └── requirements.txt
│
├── trainer/
│   ├── train.py              # PyTorch training script (DDP-ready, MNIST + CSV dataset)
│   ├── test.py               # Evaluation helper
│   ├── Dockerfile            # Trainer container
│   └── requirements.txt
│
└── infrastructure/           # (Optional) K8s manifests, helpers
```

---

## 🛠️ Requirements

- Python 3.8+
- Docker (or Minikube Docker daemon)
- Minikube (for local cluster)
- kubectl
- (Optional) AWS CLI / S3 credentials for S3 flows

Control-plane additional Python deps:
- boto3
- python-multipart

Trainer additional Python deps:
- boto3
- pandas
- Pillow

(See respective requirements.txt files under control-plane/ and trainer/)

---

## 🧭 Quick Start — Run Locally (Minikube)

1. Start Minikube
```bash
minikube start --driver=docker
```

2. Build the Trainer Docker Image
```bash
cd trainer
eval $(minikube docker-env)
docker build -t trainer:latest .
```

Verify:
```bash
minikube image ls | grep trainer
```

3. Start the FastAPI Control Plane
```bash
cd control-plane
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI:
- http://127.0.0.1:8000/docs

4. (Optional) Upload a custom dataset to S3
- POST a .zip (e.g. zipped dataset containing Training_set.csv and image folder) to:
  - `POST /datasets/upload` with multipart/form-data fields:
    - file: <zip file>
    - dataset_name: <name>
- Response: `{ "dataset_uri": "s3://bucket/path/to/your.zip" }`

5. Submit a Training Job
Use this sample JSON to POST to `/jobs`:
```json
{
  "job_name": "mnist-train-eval-1",
  "image": "trainer:latest",
  "epochs": 2,
  "dataset_size": 20000,
  "batch_size": 64,
  "world_size": 1,
  "dataset_type": "mnist"
}
```

For S3-backed custom dataset (after upload):
```json
{
  "job_name": "butterfly-run-1",
  "image": "trainer:latest",
  "epochs": 5,
  "dataset_type": "butterfly_csv",
  "dataset_uri": "s3://your-bucket/path/to/dataset.zip"
}
```

6. Monitor Training Logs
```bash
kubectl get pods
kubectl logs <pod-name>
```

Example output:
```
=== Starting PyTorch DDP-compatible training job (MNIST/custom) ===
[RANK 0] Epoch 1/2
...
[RANK 0] Model saved to /model/model.pt
[S3] Model upload complete.
=== Validation Accuracy: 85.12% ===
```

---

## 📦 Dataset Options

- MNIST (default)
  - Use `"dataset_type": "mnist"`.

- Butterfly CSV (custom S3-backed dataset)
  - Intended CSV-based dataset format:
    - A CSV named `Training_set.csv` (or consistent name) listing image filenames and labels.
    - An images folder (e.g., `train/`) inside the zip.
  - Flow:
    1. Upload zipped dataset via `POST /datasets/upload` — receives an `s3://...` URI.
    2. Submit a training job with:
       ```json
       {
         "dataset_type": "butterfly_csv",
         "dataset_uri": "s3://bucket/user-dataset.zip"
       }
       ```
    3. Trainer downloads the zip, extracts, builds train/val splits, and begins training.
  - Notes:
    - The trainer will discover the CSV and images within the extracted folder.
    - If `DATASET_URI` starts with `s3://` and `dataset_type` is `butterfly_csv`, the trainer will attempt to upload the trained model back to S3 under a models/ path related to the dataset prefix.

---

## 🐳 Trainer Dockerfile (Final Version)
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY train.py test.py /app/
RUN mkdir -p /model

CMD ["python", "/app/train.py"]
```

---

## 📡 API Reference (Control Plane)

- POST /datasets/upload
  - Description: Upload a zipped dataset to S3.
  - Request: multipart/form-data with `file` (zip) and `dataset_name` (string)
  - Response: `{ "dataset_uri": "s3://bucket/path/to/your.zip" }`
  - Environment: `S3_BUCKET` (env) must be configured for uploads (control-plane reads AWS creds from env if set)

- POST /jobs
  - Description: Submit a training job
  - Body fields:
    - `job_name` (string) — unique job name
    - `image` (string) — Docker image to run (e.g. `trainer:latest`)
    - `epochs` (int)
    - `batch_size` (int)
    - `dataset_size` (int)
    - `world_size` (int)
    - `dataset_type` (string) — `"mnist"` or `"butterfly_csv"`
    - `dataset_uri` (string, optional) — S3 URI for custom datasets

Check the FastAPI docs (`/docs`) for live schema and examples.

---

## ⚙️ Configuration / Environment

- Control plane (FastAPI)
  - Port: 8000 (configurable)
  - Kubernetes access requires configured kubeconfig (e.g., Minikube context)
  - S3 uploads require `S3_BUCKET` and AWS credentials in env to forward into job pods if needed

- Trainer container
  - Model is saved to `/model/model.pt`
  - Environment variables:
    - `EPOCHS`, `BATCH_SIZE`, `WORLD_SIZE`, `DATASET_TYPE`, `DATASET_URI`, `JOB_NAME`
  - If `DATASET_URI` is an `s3://...` path and dataset_type is `butterfly_csv`, the trainer may upload the model back to S3 under a models/ prefix.

---

## 🔧 Troubleshooting

- Pod not starting:
  - Run `kubectl describe pod <pod>` and `kubectl logs <pod>`
  - Check image name and availability in Minikube (`minikube image ls`)

- S3 upload fails:
  - Ensure `S3_BUCKET` is set for the control plane (or provide valid AWS creds in env)
  - Check IAM/credentials and region

- Distributed issues:
  - Ensure correct `MASTER_ADDR` and `MASTER_PORT` env vars are set by the job launcher
  - Verify `world_size` and per-process GPU/CPU mapping

---

## 🛣️ Roadmap

- Completed:
  - User dataset upload API → S3 (control-plane)
  - Support training on custom CSV+images dataset (butterfly_csv)
  - Model artifact upload to S3
- Future:
  - Migrate Kubernetes Jobs to AWS EKS
  - Optionally support AWS SageMaker Training Jobs
  - GPU training support

---

## ⭐ Why This Project Matters

This project demonstrates real-world ML Platform Engineering:

- Kubernetes job orchestration
- Serving a training API (like SageMaker)
- Distributed training principles
- Dockerized training environments
- Automated evaluation pipelines
- Extensible dataset architecture
- Cloud-migration-ready design (EKS, SageMaker, S3)

---

## 🤝 Contributing

Contributions are welcome! Suggested workflow:
1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Add tests (where applicable)
4. Open a PR describing your change

Please open issues for bugs or feature requests.

---

## 📜 License

This project is MIT licensed — see LICENSE for details.

---

## 👤 Author

Shail Shah  
GitHub: https://github.com/shail31-tech

---

## 📬 Contact / Support

- GitHub Issues: Open an issue in this repository for bugs or feature requests
- For quick questions, mention @shail31-tech on GitHub
