🚀 K8S Distributed Training Platform

A Mini AWS SageMaker-Style Training System Built With Kubernetes, FastAPI, and PyTorch

This project implements a complete machine learning training platform following the architecture of AWS SageMaker Training Jobs — but fully built from scratch using:

FastAPI (control plane)

Kubernetes Jobs (orchestration layer)

Docker (training runtime)

PyTorch (model training & evaluation)

Minikube (local cluster simulation)

The system supports:

✔ Distributed-compatible PyTorch training (DDP-ready)
✔ Dataset-agnostic training (MNIST or user-provided datasets)
✔ Automatic model saving and evaluation
✔ Job submission via REST API
✔ End-to-end logs and reproducible training runs
✔ Upgrade path to AWS (EKS or SageMaker)

📌 Architecture Overview
User
 ↓ (HTTP / FastAPI)
Control Plane (FastAPI)
 ↓ creates
Kubernetes Job
 ↓ pulls
Trainer Docker Image
 ↓ runs
train.py  →  saves /model/model.pt
 ↓
test.py   →  prints test accuracy
 ↓
Logs streamed back via `kubectl logs`


This mirrors the core workflow of AWS SageMaker:

User submits a training request

A job is created with hyperparameters

A container trains a model

Model artifacts + metrics are produced

🧠 Features
🔹 1. FastAPI Control Plane

POST /jobs → launch a new training job

Supports hyperparameters:

epochs

batch_size

dataset_size

world_size (DDP)

dataset_type (mnist or imagefolder)

dataset_uri (S3 support – coming soon)

🔹 2. Kubernetes-Based Job Execution

Each training request dynamically creates a Kubernetes Job, which:

Pulls the trainer:latest Docker image

Executes train.py

Trains a PyTorch CNN on MNIST (or custom data)

Saves model → /model/model.pt

Automatically runs evaluation → test.py

🔹 3. PyTorch Distributed Training Compatible

Even in single-process mode:

Uses DistributedDataParallel

Uses DistributedSampler

Training code works on CPU, GPU, single-node or multi-node

Future-proof for AWS EKS / EC2 GPU clusters.

🔹 4. Automatic Evaluation

At the end of training:

[RANK 0] Model saved to /model/model.pt
[RANK 0] Starting evaluation...
=== Test Accuracy: 97.75% ===


Evaluation is built into the same Kubernetes job — like SageMaker’s “training + evaluation script”.

📁 Project Structure
k8s-distributed-training/
│
├── control-plane/           # FastAPI service (job submission)
│   ├── app.py
│   └── ...
│
├── trainer/                 # Trainer container (PyTorch)
│   ├── train.py             # training script (DDP-ready)
│   ├── test.py              # evaluation script
│   ├── Dockerfile
│   ├── requirements.txt
│   └── ...
│
└── infrastructure/          # K8s manifests / helper scripts

🚀 How to Run Locally (Minikube)
1️⃣ Start Minikube
minikube start --driver=docker

2️⃣ Build the Trainer Image inside Minikube
cd trainer
eval $(minikube docker-env)
docker build -t trainer:latest .


Verify:

minikube image ls | grep trainer

3️⃣ Start the FastAPI Control Plane
cd control-plane
source venv/bin/activate
uvicorn app:app --reload


Open API docs:

👉 http://127.0.0.1:8000/docs

4️⃣ Submit a Training Job

Example JSON body:

{
  "job_name": "mnist-train-eval-1",
  "image": "trainer:latest",
  "epochs": 2,
  "dataset_size": 20000,
  "batch_size": 64,
  "world_size": 1
}


This triggers a Kubernetes Job.

5️⃣ Monitor Training
kubectl get pods
kubectl logs <pod-name>


Example output:

[RANK 0] --- Epoch 1/2 ---
[RANK 0] Batch 100, Avg Loss: 0.56
...
[RANK 0] Model saved to /model/model.pt
=== Test Accuracy: 97.75% ===

📦 Dataset Options
Option 1 — MNIST (default)

Just run the job with "dataset_type": "mnist".

Option 2 — User-Provided Dataset (S3-ready)

You can later allow users to:

Upload dataset → S3

Pass S3 URI to training job

Trainer downloads + unzips dataset

Uses ImageFolder for training

This makes your pipeline ready for AWS-based workloads.

⚙️ Dockerfile (Trainer)
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

🏁 Next Steps

Planned enhancements:

 Support uploading datasets to S3

 Add DATASET_URI to training Jobs

 Add automatic upload of model artifacts to S3

 Add support for multiple training backends (GPU, TPU)

 Deploy FastAPI on AWS ECS/Fargate

 Run training jobs on EKS or SageMaker Training

⭐ Why This Project Matters

This system demonstrates core ML platform engineering skills:

Kubernetes job orchestration

Dockerized ML training environments

Distributed training fundamentals (DDP)

API-driven MLOps (like SageMaker, Vertex AI, Databricks)

Scalable dataset pipeline design

Real-world ML infra thinking

This is a portfolio-grade project.

🧑‍💻 Author

Shail Shah
