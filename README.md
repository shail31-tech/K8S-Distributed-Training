# 🚀 K8S Distributed Training Platform

A Mini AWS SageMaker-Style ML Training System Using Kubernetes, FastAPI & PyTorch

## 📘 Overview

This project implements an end-to-end machine learning training platform using:

- FastAPI → Control Plane (Job Submission)
- Kubernetes Jobs → Orchestration & Execution
- Docker → Training Runtime Environment
- PyTorch → Model Training & Evaluation
- Minikube → Local K8s Cluster Simulation

It closely mirrors AWS SageMaker Training Jobs, including:

- ✔ Remote job submission
- ✔ Hyperparameter passing
- ✔ Container-based training
- ✔ Model artifact saving
- ✔ Automatic evaluation
- ✔ Distributed-training-ready code (DDP compatible)

---

## 🧠 Features

### 1. FastAPI Control Plane
- Endpoint: `POST /jobs`
- Submits training jobs to Kubernetes
- Accepts user-defined hyperparameters:
  - `epochs`
  - `batch_size`
  - `dataset_size`
  - `world_size`
  - `dataset_type` (`mnist`, or future custom dataset)
  - `dataset_uri` (S3 — future support)

### 2. Kubernetes Training Jobs
Each job:
- Pulls the `trainer:latest` Docker image
- Executes `train.py`
- Trains a PyTorch CNN (MNIST by default)
- Saves the model at `/model/model.pt`
- Automatically runs evaluation (`test.py`)
- Logs everything to `kubectl logs`

### 3. PyTorch Distributed Training Compatible
- Uses `torch.nn.parallel.DistributedDataParallel` (DDP)
- Uses `torch.utils.data.distributed.DistributedSampler`
- `world_size = 1` in Minikube (can scale to multi-GPU or AWS EKS)

### 4. Automatic Evaluation
After training completes:
```
[RANK 0] Model saved to /model/model.pt
[RANK 0] Starting evaluation...
=== Test Accuracy: 97.75% ===
```

---

## 📂 Project Structure
```
k8s-distributed-training/
│
├── control-plane/
│   ├── app.py                # FastAPI job submission service
│   └── ...
│
├── trainer/
│   ├── train.py              # PyTorch training script (DDP-ready)
│   ├── test.py               # Evaluation script
│   ├── Dockerfile            # Trainer container
│   ├── requirements.txt
│   └── ...
│
└── infrastructure/           # (Optional) K8s manifests, helpers
```

---

## 🛠️ Requirements

- Python 3.8+
- Docker (or Minikube Docker daemon)
- Minikube (for local cluster)
- kubectl
- (Optional) AWS CLI / S3 credentials for future S3 flows

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

4. Submit a Training Job

Use this sample JSON to POST to `/jobs`:
```json
{
  "job_name": "mnist-train-eval-1",
  "image": "trainer:latest",
  "epochs": 2,
  "dataset_size": 20000,
  "batch_size": 64,
  "world_size": 1
}
```

5. Monitor Training Logs
```bash
kubectl get pods
kubectl logs <pod-name>
```

Example output:
```
=== Starting PyTorch DDP-compatible training job (MNIST) ===
[RANK 0] --- Epoch 1/2 ---
...
[RANK 0] === DDP training complete ===
[RANK 0] Model saved to /model/model.pt
=== Test Accuracy: 97.75% ===
```

---

## 📦 Dataset Options

- **MNIST (default)**  
  Just use `"dataset_type": "mnist"`.

- **Custom User Dataset (S3-ready — coming soon)**  
  Planned flow:
  1. User uploads dataset to S3
  2. Receives S3 URI
  3. Calls `/jobs` with:
     ```json
     {
       "dataset_type": "imagefolder",
       "dataset_uri": "s3://bucket/user-dataset.zip"
     }
     ```
  Trainer downloads & unpacks dataset inside pod, then training begins.

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

- POST /jobs
  - Description: Submit a training job
  - Body fields:
    - `job_name` (string) — unique job name
    - `image` (string) — Docker image to run (e.g. `trainer:latest`)
    - `epochs` (int)
    - `batch_size` (int)
    - `dataset_size` (int)
    - `world_size` (int) — number of processes/NODES for DDP
    - `dataset_type` (string) — "mnist" or "imagefolder"
    - `dataset_uri` (string, optional) — S3 URI for future usage

Check the FastAPI docs (`/docs`) for live schema and examples.

---

## ⚙️ Configuration / Environment

- Control plane (FastAPI)
  - Port: 8000 (configurable)
  - Kubernetes access requires configured kubeconfig (e.g., Minikube context)

- Trainer container
  - Model is saved to `/model/model.pt` (ensure that a persistent volume is mounted if you want to keep artifacts)
  - Environment variables you may add:
    - `EPOCHS`, `BATCH_SIZE`, `WORLD_SIZE`, `DATASET_TYPE`, `DATASET_URI`

---

## 🔧 Troubleshooting

- Pod not starting:
  - Run `kubectl describe pod <pod>` and `kubectl logs <pod>`
  - Check image name and availability in Minikube (`minikube image ls`)

- GPU not detected:
  - Minikube generally runs on CPU. For GPU, use a GPU-enabled cluster or EKS with GPU nodes and a GPU-enabled base image.

- Distributed issues:
  - Ensure correct `MASTER_ADDR` and `MASTER_PORT` env vars are set by the job launcher
  - Verify `world_size` and per-process GPU/CPU mapping

---

## 🛣️ Roadmap

- User dataset upload API → S3
- Support training on custom datasets from S3
- Model artifact upload to S3
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
