"""Join C2C phrase-grounding boxes to the SDXL sample manifest."""
import argparse
import csv
import json
import re
from pathlib import Path


PATTERNS = {
    "kimono": re.compile(r"\bkimono\b", re.IGNORECASE),
    "flower": re.compile(r"\bflowers?\b", re.IGNORECASE),
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "backprop/manifest.csv")
    parser.add_argument("--records", type=Path, default=root / "concept/full_run/records")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "c2c_box_targets.csv")
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    if len(manifest) != 1200 or len({r["sample_id"] for r in manifest}) != len(manifest):
        raise ValueError("Expected 1,200 unique manifest samples")
    out_rows = []
    missing = []
    malformed = 0
    for sample in manifest:
        record_path = args.records / f"{sample['sample_id']}.json"
        if not record_path.is_file():
            missing.append(sample["sample_id"])
            continue
        record = json.loads(record_path.read_text())
        if record.get("sample_id") != sample["sample_id"]:
            raise ValueError(f"Sample ID mismatch in {record_path}")
        for concept, pattern in PATTERNS.items():
            boxes = []
            labels = []
            for source in ("grounding", "objects"):
                result = record.get(source, {})
                if len(result.get("labels", [])) != len(result.get("bboxes", [])):
                    raise ValueError(f"Label/box mismatch in {record_path}:{source}")
                for label, box in zip(result.get("labels", []), result.get("bboxes", [])):
                    if not pattern.search(str(label)):
                        continue
                    if len(box) != 4:
                        malformed += 1
                        continue
                    try:
                        x1, y1, x2, y2 = map(float, box)
                    except (TypeError, ValueError):
                        malformed += 1
                        continue
                    if not all(map(lambda x: x == x and abs(x) != float("inf"), (x1, y1, x2, y2))) or x2 < x1 or y2 < y1:
                        malformed += 1
                        continue
                    # C2C coordinates use the source image's 512 by 512 pixel space.
                    x1, x2 = max(0.0, min(512.0, x1)), max(0.0, min(512.0, x2))
                    y1, y2 = max(0.0, min(512.0, y1)), max(0.0, min(512.0, y2))
                    if x2 <= x1 or y2 <= y1:
                        malformed += 1
                        continue
                    boxes.append([round(x1 / 512, 7), round(y1 / 512, 7), round(x2 / 512, 7), round(y2 / 512, 7)])
                    labels.append({"source": source, "label": str(label)})
            out_rows.append({
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "race_prompt": sample["race_prompt"],
                "gender_prompt": sample["gender_prompt"],
                "concept": concept,
                "detector_present": int(bool(boxes)),
                "boxes_xyxy_normalized": json.dumps(boxes, separators=(",", ":")),
                "box_sources_labels": json.dumps(labels, separators=(",", ":")),
            })
    if missing:
        raise ValueError(f"Missing {len(missing)} C2C records; first IDs: {missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0])
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} sample/concept targets; discarded {malformed} malformed boxes: {args.output}")


if __name__ == "__main__":
    main()
