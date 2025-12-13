from typing import Dict, Optional
from kubernetes import client, config

config.load_kube_config()
batch_v1 = client.BatchV1Api()

def build_job_manifest(
    job_name: str,
    image: str = "trainer:latest",
    env: Optional[Dict[str, str]] = None,
):
    env_list = []
    if env:
        env_list = [{"name": k, "value": str(v)} for k, v in env.items()]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "trainer",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "env": env_list,
                        }
                    ],
                    "restartPolicy": "Never",
                }
            }
        },
    }

def create_job(job_name: str, image: str = "trainer:latest", env: Optional[Dict[str, str]] = None):
    manifest = build_job_manifest(job_name, image, env)
    return batch_v1.create_namespaced_job(namespace="default", body=manifest)




def get_job_status(job_name: str):
    job = batch_v1.read_namespaced_job_status(job_name, namespace="default")
    return job.status
