#!/usr/bin/env bash
# 知枢 · 一键复现脚本
# 用法： bash scripts/reproduce.sh          # 降级模式（无需任何外部服务/Key）
#        bash scripts/reproduce.sh --real   # 真实模式（需 .env 配好百炼 Key）
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="degrade"
if [[ "${1:-}" == "--real" ]]; then MODE="real"; fi

echo "=============================================="
echo " 知枢 复现模式：$MODE"
echo "=============================================="

if [[ "$MODE" == "degrade" ]]; then
  export VECTOR_BACKEND=local
  export METADATA_BACKEND=sqlite
  export CACHE_BACKEND=memory
  export EMBED_PROVIDER=hash
  export LLM_API_KEY=""
  export ENABLE_TRACE=false
  echo "[提示] 降级模式：本地向量库 + SQLite + 内存缓存 + mock 模型"
else
  echo "[提示] 真实模式：读取 .env 中的百炼 Key，连接 Milvus/PG/Redis"
  if [[ ! -f .env ]]; then echo "[错误] 缺少 .env，请先 cp .env.example .env 并填 Key"; exit 1; fi
fi

echo ""
echo ">>> 1/4 单元测试"
python -m pytest tests/ -q

echo ""
echo ">>> 2/4 端到端演示（入库 → 检索 → 生成 → 引用校验）"
python -m scripts.demo --rebuild

echo ""
echo ">>> 3/4 评测"
if [[ "$MODE" == "real" ]]; then
  python -m scripts.run_eval --gate
else
  python -m scripts.run_eval --no-judge
fi

echo ""
echo ">>> 4/4 完成。trace 见 data/traces/，报告见 data/reports/"
