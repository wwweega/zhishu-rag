#!/usr/bin/env python3
"""
知枢 · 批量入库脚本
用法：
  python -m scripts.ingest                     # 入库 data/corpus 下全部文档
  python -m scripts.ingest path/to/a.pdf       # 入库指定文件
  python -m scripts.ingest --dir docs/ --dept 人力资源
  python -m scripts.ingest --rebuild           # 先清空知识库再入库（避免历史残留）

面试考点：
  - 入库是**幂等**的（doc_id 基于路径哈希），重复入库同一文件是"覆盖"不是"追加"；
  - 但"删除旧文档"只删向量不删实体文件的场景下，切换语料仍可能残留，
    因此提供 --rebuild 走"清空 + 重建"的干净路径，便于复现实验。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import get_pipeline  # noqa: E402

SUPPORTED = {".pdf", ".md", ".txt", ".csv", ".docx"}


def collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files = [p for p in sorted(target.rglob("*")) if p.suffix.lower() in SUPPORTED]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="知枢 批量入库")
    ap.add_argument("paths", nargs="*", default=["data/corpus"], help="文件或目录")
    ap.add_argument("--dir", default=None, help="目录（等价于位置参数传目录）")
    ap.add_argument("--dept", default="", help="归属部门（用于元数据过滤）")
    ap.add_argument("--rebuild", action="store_true", help="先清空知识库再入库")
    args = ap.parse_args()

    targets = [Path(p) for p in (args.paths or ["data/corpus"])]
    if args.dir:
        targets.append(Path(args.dir))

    files: list[Path] = []
    for t in targets:
        files.extend(collect(t))
    files = [f for f in files if f.exists()]

    if not files:
        print(f"[警告] 没有找到可入库文件：{[str(t) for t in targets]}")
        print(f"        支持的后缀：{sorted(SUPPORTED)}")
        return 1

    pipe = get_pipeline()
    if args.rebuild:
        print("[重建] 清空知识库 ...")
        pipe.vector_store.drop()
        pipe.meta.clear()
        pipe.refresh_index()

    print(f"[入库] 共 {len(files)} 个文件")
    ok = 0
    for f in files:
        r = pipe.ingest(f, dept=args.dept)
        if r.get("ok"):
            ok += 1
            print(f"  ✓ {f.name:<40} 解析={r['parser']:<8} "
                  f"子块={r['children']:<4} 父块={r['parents']:<3} "
                  f"{r['elapsed_ms']}ms  trace={r['trace_id']}")
            for w in r.get("warnings", [])[:2]:
                print(f"      ⚠ {w}")
        else:
            print(f"  ✗ {f.name:<40} 失败：{r.get('error')}")

    print(f"\n[完成] 成功 {ok}/{len(files)}")
    print(json.dumps(pipe.stats(), ensure_ascii=False, indent=2, default=str))
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    sys.exit(main())
