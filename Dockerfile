# Dockerfile
FROM python:3.8-slim  

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OpenCV需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY pyproject.toml .

# 安装Python依赖（包括DVC、MLflow）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 初始化DVC（确保容器内可用）
RUN dvc init --no-scm

# 设置环境变量（从.env加载，或在docker-compose中覆盖）
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# 暴露端口（APP运行端口）
EXPOSE 8000

# 默认命令（运行APP主入口）
CMD ["python", "app/main.py", "--help"]