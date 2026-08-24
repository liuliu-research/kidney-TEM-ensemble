"""生成不可变协议合同；``--dry-run`` 严格只读。"""

from __future__ import annotations

import argparse
import json

from strict_oof.protocol import (
    atomic_write_json,
    build_cache_inventory,
    build_contract,
    ensure_runtime_dirs,
    paths,
)
from strict_oof.preflight import ensure_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    p = paths()
    if not args.dry_run:
        ensure_runtime_dirs(p)
        build_cache_inventory(p.exp)
        ensure_preflight()
    candidate = build_contract(p.exp)
    summary = {
        key: candidate[key]
        for key in (
            "protocol_id",
            "contract_sha256",
            "train_images",
            "metacal_images",
            "test_images",
            "train_patients",
            "metacal_patients",
            "test_patients",
            "source_code_sha256",
        )
    }
    summary["fold_records"] = candidate["fold_records"]
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if p.contract.exists():
        existing = json.loads(p.contract.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") != candidate["contract_sha256"]:
            raise RuntimeError(
                "已有严格 OOF 合同与当前源码/数据不同；" "禁止覆盖并混用产物"
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    atomic_write_json(p.contract, candidate)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
