from __future__ import annotations

import csv

import pytest

from index_topk_perflab.profile_validation import validate_ncu_csv, validate_nsys_stats


def test_validate_ncu_csv_requires_all_requested_metrics(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    label = "ITK::STAGE::candidate::case::topk"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Kernel Name",
                "NVTX",
                "gpu__time_duration.sum",
                "dram__bytes_read.sum",
                "lts__t_sectors_op_read.sum",
            ]
        )
        writer.writerow(["kernel", label, "12.5", "1024", "32"])

    result = validate_ncu_csv(
        path,
        requested_metrics=("dram__bytes_read.sum", "lts__t_sectors_op_read.sum"),
        expected_nvtx_label=label,
        stage="topk",
    )
    assert result["status"] == "passed"
    assert result["kernel_rows"] == 1
    assert result["requested_metric_finite_value_counts"] == {
        "dram__bytes_read.sum": 1,
        "lts__t_sectors_op_read.sum": 1,
    }


def test_validate_ncu_csv_rejects_missing_metric(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(
        '"Kernel Name","NVTX","gpu__time_duration.sum"\n'
        '"kernel","ITK::STAGE::v::c::topk","1"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing requested metrics"):
        validate_ncu_csv(
            path,
            requested_metrics=("dram__bytes_read.sum",),
            expected_nvtx_label="ITK::STAGE::v::c::topk",
            stage="topk",
        )


def test_validate_nsys_stats_requires_full_nvtx_label(tmp_path) -> None:
    path = tmp_path / "stats.csv"
    label = "ITK::RUN::candidate::prefill-profile-q4096-kv131072-k2048"
    path.write_text(f'Range,Kernel,Time\n"{label}",kernel,10\n', encoding="utf-8")
    result = validate_nsys_stats(path, expected_nvtx_label=label)
    assert result["nvtx_label_occurrences"] == 1

    with pytest.raises(ValueError, match="do not contain NVTX label"):
        validate_nsys_stats(path, expected_nvtx_label=f"{label}-missing")
