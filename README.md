🚀 K8S Distributed Training Platform

A Mini AWS SageMaker-Style ML Training System Using Kubernetes, FastAPI & PyTorch

📘 Overview

This project implements an end-to-end machine learning training platform using:

FastAPI → Control Plane (Job Submission)

Kubernetes Jobs → Orchestration & Execution

Docker → Training Runtime Environment

PyTorch → Model Training & Evaluation

Minikube → Local K8s Cluster Simulation

It closely mirrors AWS SageMaker Training Jobs, including:

✔ Remote job submission
✔ Hyperparameter passing
✔ Container-based training
✔ Model artifact saving
✔ Automatic evaluation
✔ Distributed-training-ready code (DDP compatible)

🧠 Features
1. FastAPI Control Plane

Endpoint: POST /jobs

Submits training jobs to Kubernetes

Accepts user-defined hyperparameters:

epochs

batch_size

dataset_size

world_size

dataset_type (mnist, or future custom dataset)

dataset_uri (S3 — future support)

2. Kubernetes Training Jobs

Each job:

Pulls the trainer:latest Docker image

Executes train.py

Trains a PyTorch CNN (MNIST by default)

Saves the model at /model/model.pt

Automatically runs evaluation (test.py)

Logs everything to kubectl logs

3. PyTorch Distributed Training Compatible

Uses DistributedDataParallel (DDP)

Uses DistributedSampler

World size = 1 in Minikube (can scale to multi-GPU or AWS EKS)
