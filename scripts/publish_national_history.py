"""Publish only the verified, historical aircraft records used by the workbench."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("evaluation/results/aircraft_v2")
CSV = EVIDENCE / "remotesam_aircraft_all_experiments_verified.csv"
SUMMARY = EVIDENCE / "remotesam_aircraft_experiment_summary_verified.json"
OUTPUT = ROOT / "web/data/national-history.json"
BASELINE = "30306a66ce4ad95e49922f1ec5415bc5d8b9004b"
AFR_COMMIT = "c1ef2259bb2148be93ace13d55cf1227540cb554"
AFR_ARCHIVE = Path("evaluation/national_final/results/lab_runs/20261117_ablation")
AFR_RUNS = {
    "A3_budget0.25_lam1.0": "a3_b025_l1",
    "A3_budget0.5_lam1.0": "a3_b050_l1",
    "A4_budget0.25_lam0.3": "a4_b025_l03",
    "A4_budget0.25_lam0.5": "a4_b025_l05",
    "A4_budget0.25_lam0.7": "a4_b025_l07",
    "A5_budget0.25_lam0.7": "a5_b025_l07",
}
GROUPS = {
    "A0": "prompt_all_airplanes",
    "A1": "prompt_all_planes",
    "A2": "tile_1024_overlap_256_p0_88",
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def read_afr_archive(path):
    return subprocess.check_output(
        ["git", "show", f"{AFR_COMMIT}:{(AFR_ARCHIVE / path).as_posix()}"], cwd=ROOT
    ).decode("utf-8-sig")


def publish_afr():
    audit = json.loads(read_afr_archive("audit.json"))
    summary = json.loads(read_afr_archive("results/ablation_summary.json"))
    version = audit["service"]["version"]
    checkpoint = audit["service"]["checkpoint_sha256"]
    with (ROOT / "evaluation/dataset_manifest.csv").open(encoding="utf-8", newline="") as handle:
        expected_ids = {row["image_id"] for row in csv.DictReader(handle) if row["class_name"] == "plane"}
    if audit["scope"] != "historical_diagnostic" or len(expected_ids) != 10:
        raise ValueError("Unexpected AFR data scope")

    records = []
    for key, filename in AFR_RUNS.items():
        result = summary[key]
        config = result["config"]
        csv_path = f"results/{filename}.csv"
        rows = list(csv.DictReader(read_afr_archive(csv_path).splitlines()))
        if (result["status"] != "completed" or result["threshold_validated"] is not False
                or len(rows) != result["rows_total"] or len(rows) != result["rows_succeeded"]
                or len(rows) != len(expected_ids) or {row["image_id"] for row in rows} != expected_ids):
            raise ValueError(f"{key}: unexpected evaluation coverage")
        for row in rows:
            if (row["status"] != "success" or row["service_version"] != version
                    or row["checkpoint_sha256"] != checkpoint
                    or row["config_sha256"] != config["config_sha256"]):
                raise ValueError(f"{key}: inconsistent status, checkpoint or configuration")

        metrics = {}
        columns = {
            "iou": "iou", "dice": "dice", "precision": "precision", "recall": "recall",
            "inference_ms": "service_inference_ms",
        }
        for metric, column in columns.items():
            recorded = result[f"mean_{column}"]
            values = [float(row[column]) for row in rows if row[column] != ""]
            if not values or (len(values) != len(rows) and metric != "precision"):
                raise ValueError(f"{key}: missing {column} values")
            computed = mean(values)
            tolerance = 0.001 if metric == "inference_ms" else 0.000001
            if abs(recorded - computed) > tolerance:
                raise ValueError(f"{key}: {column} differs from the per-image CSV")
            metrics[metric] = recorded

        scale = " · 局部放大" if config["upscale"] else ""
        records.append({
            "group": result["experiment"],
            "status": "completed",
            "real_execution": True,
            "samples": len(rows),
            "precision_samples": sum(bool(row["precision"]) for row in rows),
            "dataset_role": "iSAID/DOTA 飞机历史诊断集",
            "config": (
                f'GeoScope-AFR {config["algorithm_version"]} · RemoteSAM {version} · '
                f'预算 {config["budget_fraction"]:.0%} · λ {config["lam"]:.1f} · 阈值 {config["threshold"]} · '
                f'{config["tile_size"]} 切片 / 重叠 {config["overlap"]} px{scale}'
            ),
            "metrics": metrics,
            "artifact_url": (
                f"https://github.com/SitianGao/geoscope/blob/{AFR_COMMIT}/{(AFR_ARCHIVE / csv_path).as_posix()}"
            ),
            "source_experiment_id": filename,
        })
    write_records(ROOT / "web/data/national-experiments.json", records)
    print(f"Published {len(records)} AFR configurations from {AFR_COMMIT[:7]}")


def publish():
    summary = json.loads((ROOT / SUMMARY).read_text(encoding="utf-8"))
    with (ROOT / CSV).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    records = []
    for group, experiment in GROUPS.items():
        values = summary["experiments"][experiment]
        selected = [row for row in rows if row["experiment_id"] == experiment]
        if len(selected) != values["images"] or any(row["status"] != "success" for row in selected):
            raise ValueError(f"{experiment}: historical evidence no longer matches its summary")
        counts = [(int(row["tp"]), int(row["fp"]), int(row["fn"])) for row in selected]
        precision = mean(ratio(tp, tp + fp) for tp, fp, fn in counts)
        recall = mean(ratio(tp, tp + fn) for tp, fp, fn in counts)

        config = values["configuration"]
        method = "整图"
        if config["strategy"] == "tile":
            method = f'{config["tile_size"]} 切片 · 重叠 {config["overlap"]} · Hann 加权'
        records.append({
            "group": group,
            "status": "completed",
            "real_execution": True,
            "samples": values["images"],
            "dataset_role": "iSAID/DOTA 飞机历史诊断集",
            "config": (
                f'RemoteSAM {summary["service_version"]} · {config["prompt"]} · '
                f'{method} · 阈值 {config["probability_threshold"]}'
            ),
            "metrics": {
                "iou": values["mean_iou_failure_as_zero"],
                "dice": values["mean_dice_failure_as_zero"],
                "precision": precision,
                "recall": recall,
                "inference_ms": values["mean_service_inference_ms_per_image"],
            },
            "artifact_url": f"https://github.com/SitianGao/geoscope/blob/{BASELINE}/{CSV.as_posix()}",
            "source_experiment_id": experiment,
        })

    write_records(OUTPUT, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-afr", action="store_true", help="Read the pinned AFR archive from local Git objects")
    args = parser.parse_args()
    publish()
    if args.include_afr:
        publish_afr()
