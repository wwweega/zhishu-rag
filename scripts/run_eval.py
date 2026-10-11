#!/usr/bin/env python3
"""
知枢 · 评测入口
用法：
  python -m scripts.run_eval                 # 全量评测（含 LLM judge）
  python -m scripts.run_eval --limit 6       # 只跑前 6 题（快速验证）
  python -m scripts.run_eval --gate          # 跑完并执行 CI 门禁（不达标退出码非 0）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.harness import check_gate, run_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="知枢 RAG 评测")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判")
    ap.add_argument("--gate", action="store_true", help="执行 CI 门禁")
    args = ap.parse_args()

    summary = run_eval(limit=args.limit, use_judge=not args.no_judge)
    print(json.dumps({k: v for k, v in summary.items() if k != "details"},
                     ensure_ascii=False, indent=2))

    if args.gate:
        ok, failures = check_gate(summary)
        if not ok:
            print("\n[门禁未通过]")
            for f in failures:
                print("  ✗", f)
            return 1
        print("\n[门禁通过] 所有指标达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
