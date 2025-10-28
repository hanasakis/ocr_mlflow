# Dockerfile
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（针对Debian bookworm修正）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件（利用Docker缓存层）
COPY requirements.txt .
COPY pyproject.toml .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV PYTHONPATH=/app/src

# 创建必要的目录
RUN mkdir -p /app/data /app/models /app/logs

# 暴露端口
EXPOSE 8000

# 设置默认命令（显示帮助信息）
CMD ["python", "app/main.py", "--help"]