# 贡献指南

感谢关注知枢。本项目的目标是做一个**可复现、可验证、可讲解**的企业级 RAG 参考实现，
所以对"正确性"和"可解释性"的要求高于对"功能数量"的追求。

## 开发环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q          # 34 个测试，默认全走降级后端
```

无需任何外部服务即可跑通全部测试与 demo（Milvus/PG/Redis/模型 Key 均可缺省降级）。

## 提交前自检

```bash
python -m pytest tests/ -q                  # 单测必须全绿
python -m scripts.run_eval --gate           # 质量门禁必须通过（需模型 Key）
```

若你没有模型 Key，请至少保证单测全绿，并在 PR 说明中注明"未跑真实评测"。

## 代码约定

1. **处处可降级**：新增外部依赖时，必须提供"依赖缺失时也能跑"的降级路径，
   并在 `app/storage/*.py` 或 `app/core/llm_client.py` 的工厂函数里注册。
2. **关键逻辑写"面试考点"注释**：本项目同时是学习材料，复杂决策处请说明
   *为什么这么选*、*不这么选会怎样*。
3. **修 bug 必须补回归测试**：本项目刻意保留了一份"踩坑记录"（见 README 第六节），
   每个坑都有对应测试，防止回退。
4. **不造数**：README / 文档里出现的任何指标，必须来自真实运行，禁止示例化编造。

## 目录

- `app/core` 配置 / LLM 客户端 / 可观测
- `app/storage` 向量库 / 元数据 / 缓存（均双后端）
- `app/ingest` 解析 / 分块
- `app/retrieval` 检索 / 融合 / 改写
- `app/generation` 重排 / 生成治理 / 引用校验
- `app/eval` 评测与门禁
- `tests/` 单测（含全部回归测试）

## 提交信息规范

采用轻量 Conventional Commits：

```
feat(retrieval): 支持过滤条件下推到 Milvus expr
fix(cache): 版本号改存 Redis，修复多进程脏读
docs(readme): 补充真实的评测结果
test(observability): 补嵌套 span 阶段耗时并集回归测试
```

## 行为准则

- 对事不对人，讨论聚焦技术方案；
- 欢迎提出反例与质疑 —— 本项目的很多修复正来自"这个数好像不对"的质疑。
