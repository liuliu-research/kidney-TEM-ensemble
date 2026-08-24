"""深验收全绿后原子发布严格 OOF 指针，不覆盖旧结果。"""

from __future__ import annotations

import json

from strict_oof.delivery import build_delivery, refresh_payload_manifest
from strict_oof.protocol import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_contract,
    paths,
)


def main() -> int:
    contract = load_contract(fresh=True)
    p = paths()
    from strict_oof.validate import validate_marker, validate_step

    marker_errors = validate_marker("validate_strict_oof", validate_artifact=True)
    if marker_errors:
        raise RuntimeError(f"深验收marker无效，禁止发布: {marker_errors}")
    # finalize marker丢失时允许对健康READY原样复用，避免重跑core validator的
    # checked_at改变哈希而破坏幂等性；编排器随后只需重新发布marker。
    if p.ready.is_file():
        existing_errors = validate_step(
            "finalize_strict_oof", deep=True, contract_check=False
        )
        if not existing_errors:
            existing = json.loads(p.ready.read_text(encoding="utf-8"))
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return 0
        raise RuntimeError(
            "existing READY is invalid and must be quarantined before rebuild: "
            f"{existing_errors}"
        )
    report = json.loads(p.validation_report.read_text(encoding="utf-8"))
    if (
        report.get("ok") is not True
        or report.get("status") != "complete"
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("errors")
    ):
        raise RuntimeError("深验收报告未全绿，禁止发布")
    analysis = p.out / "analysis.json"
    ablation = p.out / "extended_ablation_analysis.json"
    for target in (analysis, ablation):
        if not target.is_file():
            raise RuntimeError(f"缺少发布产物: {target}")
    delivery = build_delivery(contract)
    # Composite 目录上先运行 strict-aware core validator；只有全绿才
    # 发布现有本地接续器识别的三道门。
    import validate_core_results as core_validator

    core_report = core_validator.validate(delivery, p.big)
    if core_report.get("ok") is not True or core_report.get("errors"):
        raise RuntimeError(
            "composite delivery core validation failed: " f"{core_report.get('errors')}"
        )
    atomic_write_json(delivery / "core_validation.json", core_report)
    report_sha = file_sha256(delivery / "core_validation.json")
    analysis_sha = file_sha256(delivery / "analysis.json")
    marker_text = (
        f"report_sha256={report_sha}\n"
        f"analysis_sha256={analysis_sha}\n"
        f"contract_sha256={contract['contract_sha256']}\n"
        "validation_mode=strict_oof_composite_delivery\n"
    )
    atomic_write_text(delivery / "markers" / "core_validation.done", marker_text)
    atomic_write_text(
        delivery / "markers" / "ALL.done",
        f"strict_oof_complete\ncontract_sha256={contract['contract_sha256']}\n",
    )
    refresh_payload_manifest(delivery, contract)
    ready = {
        "status": "ready",
        "contract_sha256": contract["contract_sha256"],
        "validation_report_sha256": file_sha256(p.validation_report),
        "analysis_sha256": file_sha256(analysis),
        "extended_ablation_analysis_sha256": file_sha256(ablation),
        "strict_results_dir": str(p.out),
        "strict_big_dir": str(p.big),
        "delivery_results_dir": str(delivery),
        "strict_results_project_relative_path": p.out.relative_to(
            p.exp.parent
        ).as_posix(),
        "strict_big_project_relative_path": p.big.relative_to(p.exp.parent).as_posix(),
        "delivery_project_relative_path": delivery.relative_to(p.exp.parent).as_posix(),
        "delivery_core_validation_sha256": report_sha,
        "delivery_analysis_sha256": analysis_sha,
        "delivery_payload_manifest_sha256": file_sha256(
            delivery / "payload_manifest.json"
        ),
        "legacy_results_overwritten": False,
        "activation_semantics": (
            "This pointer publishes a complete isolated composite results "
            "view that can be passed directly as --results-dir. Legacy "
            "results/checkpoints remain unchanged."
        ),
    }
    atomic_write_json(p.ready, ready)
    active_pointer = p.exp / "results" / "strict_oof_ACTIVE.json"
    atomic_write_json(active_pointer, ready)
    final_errors = validate_step("finalize_strict_oof", deep=True, contract_check=False)
    if final_errors:
        raise RuntimeError(f"finalize self-validation failed: {final_errors}")
    print(json.dumps(ready, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
