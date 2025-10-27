# CNN OCR数字识别项目

一个基于CNN的手写数字OCR识别项目，支持单数字和多数字识别。

## 功能特性

- 🎯 高精度手写数字识别
- 📷 支持单数字和多数字识别
- 🔧 可配置的模型参数
- 🐳 Docker容器化支持
- ✅ 完整的CI/CD流水线
- 🧪 单元测试覆盖

## 项目结构
# ocr_mlflow


## DevOps DevOps Workflow

### 1. Version Control Strategy

#### 1.1 Branch Structure & Responsibilities
| Branch Name    | Responsibility                          | Protection Level | Merge Source       | Merge Target       |
|----------------|-----------------------------------------|------------------|--------------------|--------------------|
| `main`         | Production-ready code (stable deployable) | Highest          | `staging`          | None (receives merges only) |
| `staging`      | Pre-production code (after testing)     | High             | `dev`              | `main`             |
| `dev`          | Development code (feature integration)  | Medium           | `feature/*`        | `staging`          |
| `feature/*`    | Feature branches (e.g., `feature/dvc-setup`) | Low        | None (created from `dev`) | `dev`       |


#### 1.2 Code Submission Workflow
1. **Create Feature Branch**: Pull latest code from `dev` and create a `feature/xxx` branch
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/add-multi-predict

   Local Development & Validation: After development, validate with pre-commit checks and tests
   pre-commit run --all-files  # Code format/lint checks
pytest tests/ -v            # Unit tests
python app/main.py tests/mnist_style.png --mode single  # Functional validation

Commit & Push: Submit code to remote feature branch and create a PR to dev

git add .
git commit -m "feat: add multi-digit predict function"
git push origin feature/add-multi-predict

Merge Validation: PR triggers CI/CD pipeline (code checks→DVC pull→testing). Merge to dev after approval.
Pre-production & Production: After integration in dev, create PR to staging (pre-production testing), then merge to main (production).

2. CI/CD Automated Pipeline (GitHub Actions)
2.1 Pipeline Trigger Rules
Event	Target Branches	Tasks Executed
PR submission/update	dev/staging/main	Code checks (pre-commit)→DVC pull→unit tests
Code merge	dev	Code checks→testing→model training (MLflow logged)
Code merge	staging	Code checks checks→testing→training→build pre-production Docker image
Code merge	main	Code checks→test→training→build production Docker image
2.2 Core Pipeline Steps (e.g., for staging merge)
Code Checkout: Pull latest code from the target branch
Environment Setup: Install Python dependencies, configure DVC/MLflow (using GitHub Secrets)
Code Quality Checks: Run black (formatting), flake8 (linting), mypy (type checking)
Data Retrieval: Pull latest data versions via DVC from DAGsHub remote storage
Testing Execution: Run unit tests and functional tests for OCR core functions
Model Training: Trigger automated training (if merged to dev/staging), log metrics to MLflow
Docker Image Build: Create Docker image tagged with branch name (e.g., ocr-app:staging-latest)
Validation: Test the built image to ensure runtime functionality
3. Docker Deployment
Local Testing: Use docker-compose to run the app or training service
bash
# Run OCR app
docker-compose up ocr-app

# Run model training (with DVC data pull)
docker-compose --profile training up
Environment Configuration:
Dockerfile: Optimized for production (slim base image, minimal dependencies)
docker-compose.yml: Separates app and training services with shared volumes for data/models
Environment variables injected via .env (see .env.example for template)
4. Key Tools & Integrations
Version Control: Git + GitHub (branch management, PR reviews)
Data Versioning: DVC + DAGsHub (large file tracking, remote storage)
Experiment Tracking: MLflow + DAGsHub (metrics, hyperparameters, model artifacts)
CI/CD: GitHub Actions (automated testing, training, and deployment)
Containerization: Docker + docker-compose (environment consistency)
Code Quality: pre-commit hooks (black, flake8, mypy)
