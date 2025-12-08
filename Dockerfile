# Dockerfile
FROM python:3.10-slim

# 安装系统依赖（ffmpeg 等）
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 复制 requirements
COPY requirements.txt /app/requirements.txt

# 安装 Python 依赖（不包含 demucs）
RUN pip install --no-cache-dir -r /app/requirements.txt

# 安装 demucs（最新版本）
RUN pip install --no-cache-dir demucs

# 复制应用代码
COPY app /app/app

# 创建输出目录
RUN mkdir -p /app/outputs
VOLUME ["/app/outputs"]

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
