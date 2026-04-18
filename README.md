# 🍅 Tomato Leaf AI: Production-Grade MLOps Platform


End-to-end MLOps system for real-time tomato leaf disease detection using a distributed, asynchronous Kubernetes architecture.  
This project demonstrates the transition from a standalone model to a scalable inference system featuring background task processing, deterministic model routing, and real-time environmental drift observability.  

**Author:** Arindam Das (M.E., Jadavpur University)

---

## 🏗️ System Architecture & Design Patterns

The platform utilizes a Decoupled Gateway-Worker pattern to isolate synchronous web traffic from asynchronous, heavy-compute AI tasks.

---

### 1. High-Performance Traffic Gateway (Nginx)

The entry point is a hardened Unprivileged Nginx Reverse Proxy optimized for AI workloads:

- AI-Optimized Load Balancing: Implements the least_conn algorithm to route traffic to the least busy API pod, optimizing for varying AI inference times.
- Layer 7 Rate Limiting: Employs a "Leaky Bucket" strategy (10r/s with a burst of 20) to protect the cluster from DDoS and script-based abuse.
- Startup Resilience: Utilizes a Kubernetes InitContainer to perform DNS lookups, preventing startup race conditions by waiting until the API service is resolvable.
- Non-Root Hardening: Runs on port 8080 using the nginxinc/nginx-unprivileged image. The configuration redirects the PID file and temporary buffers to /tmp to support a readOnlyRootFilesystem.

---

### 2. Asynchronous API Gateway (FastAPI)

- Memory Protection: Strict 10MB Payload Limit check enforced at the application layer before reading files into memory, preventing RAM exhaustion attacks.
- Headless Service Integration: Deployed with clusterIP: None, allowing the Nginx gateway to perform true peer-to-peer load balancing across individual Pod IPs.
- Dependency Management: Uses an InitContainer to verify Redis availability before the main application process initializes.

---

### 3. Distributed AI Worker Cluster (Celery & TensorFlow)

#### Two-Stage Staged Inference:

- Stage 1 (Gatekeeper): A lightweight MobileNetV2 validates if the image is a tomato leaf.
- Stage 2 (Classifier): Heavy models (ResNet50/EfficientNet) only run on valid data, reducing unnecessary inference on invalid inputs (~60%).

#### Deterministic A/B Testing:

- Uses MD5 hashing on user_id to consistently route 70% of traffic to EfficientNetB0 and 30% to ResNet50.

#### Fault Tolerance:

- Configured with Exponential Backoff Retries and task_acks_late=True to minimize risk of task loss during pod evictions.

---

## 🚀 Advanced MLOps Features

### 📈 Event-Driven Autoscaling (KEDA)

- Workers scale from 2 to 5 replicas based on Redis Queue Depth via KEDA. This ensures the cluster reacts to actual workload (pending images) rather than secondary metrics like CPU, maintaining high availability during traffic bursts.

---

### 🔍 Environmental Data Drift Monitoring

The platform tracks Statistical Integrity in real-time by comparing incoming data against the PlantVillage Training Baseline (18,339 images):

- Metrics: Calculates real-time Brightness and Contrast distributions.
- Alerting: If a user uploads an image that is too dark or blurry, the system flags it as "Environmental Drift" via Grafana gauges, signaling potential accuracy degradation.

---

### 🧹 Automated Storage Hygiene

- A Kubernetes CronJob runs hourly to perform garbage collection on the Persistent Volume, purging abandoned upload sessions older than 2 hours to ensure storage availability.

---

## 🛡️ Security & Infrastructure Hardening

### 1. Zero-Trust Network Security

Strict Kubernetes NetworkPolicies enforce a "Least Privilege" communication model:

- API Firewall: Only Nginx pods are permitted to communicate with the API.
- Redis Firewall: Only API and Worker pods can access the Redis broker.
- Database Firewall: Only Worker and MLflow pods can reach the Postgres database.
- Monitoring Firewall: Prometheus is restricted to scraping specific metrics endpoints across the namespace.

---

### 2. Container Hardening

- readOnlyRootFilesystem: true: Hardens containers against runtime code modification or malware installation.
- Non-Root Execution: All processes run with non-root UIDs (1000 or 101).
- Stateful Integrity: Databases (Redis/Postgres) and Monitoring (Prometheus/MLflow) use StatefulSets to ensure stable network IDs and data persistence across restarts.

---

## 📡 API Specification

| Endpoint                    | Method | Technical Description |
|---------------------------|--------|----------------------|
| /upload                   | POST   | Validates metadata & triggers asynchronous Gatekeeper task. |
| /leaf_checker/{uid}/{tid} | GET    | Polls Redis for Stage 1 validation status. |
| /predict/{uid}            | POST   | Dispatches classification task to the distributed worker pool. |
| /result/{uid}/{tid}       | GET    | Retrieves inference results; session cleanup handled asynchronously/cron-based. |

---

## 📊 Observability Stack

- Grafana: Real-time visualization of throughput, A/B test distribution, and data drift gauges.
- Prometheus & Pushgateway: Multi-tier metrics collection with persistent storage.
- MLflow: Centralized Model Registry and Audit Trail, logging every inference with its associated drift statistics and latency.

---

## 🙏 Acknowledgements & Development Approach

This project was developed using modern AI-assisted development tools, including Google AI Studio, for implementation support and rapid iteration.  

The system architecture, Kubernetes orchestration, MLOps pipeline design, and overall integration were designed, implemented, and debugged by the author.  

This project reflects a hands-on effort to build and understand a production-grade distributed AI system from end to end.

---

## 📜 License

Distributed under the MIT License. See LICENSE for more information.

---

## ✉️ Contact

Arindam Das - darindam.das07@gmail.com  

LinkedIn: https://www.linkedin.com/in/arindam-das-7a99a225a  

GitHub: https://github.com/ArindamDas07
