"""Publish only the verified, historical aircraft records used by the workbench."""
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("evaluation/results/aircraft_v2")
CSV = EVIDENCE / "remotesam_aircraft_all_experiments_verified.csv"
SUMMARY = EVIDENCE / "remotesam_aircraft_experiment_summary_verified.json"
OUTPUT = ROOT / "web/data/national-history.json"
BASELINE = "30306a66ce4ad95e49922f1ec5415bc5d8b9004b"
GROUPS = {
    "A0": "prompt_all_airplanes",
    "A1": "prompt_all_planes",
    "A2": "tile_1024_overlap_256_p0_88",
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    publish()
