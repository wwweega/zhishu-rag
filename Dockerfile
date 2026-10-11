# ============================================================
# 知枢 · 多阶段构建
# 面试考点：为什么要多阶段构建？
#   → 构建期需要 gcc 等工具链编译 wheel，运行期完全不需要。
#     多阶段能把镜像从 ~1.2GB 降到 ~400MB，同时**减小攻击面**
#     （运行镜像里没有编译器 = 少了被利用的入口）。
# ============================================================
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
# 中文字体与 PDF 依赖（PyMuPDF 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p data/corpus data/index data/traces data/reports
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8080/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
