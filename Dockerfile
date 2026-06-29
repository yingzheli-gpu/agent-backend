# 后端 Dockerfile - 多智能体中医问诊系统
# 多阶段构建优化镜像大小

# ============================================
# 构建阶段 - 安装依赖
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml ./

# 安装 uv 包管理器
RUN pip install --no-cache-dir uv

# ============================================
# 运行阶段 - 最终镜像
# ============================================
FROM python:3.11-slim

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 包管理器
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml ./

# 使用 uv 安装依赖
RUN uv pip install --system -e .

# 复制源代码
COPY . .

# 创建日志目录
RUN mkdir -p logs

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "main.py"]
