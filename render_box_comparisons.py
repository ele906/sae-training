"""Render C2C target and SAE predicted boxes over held-out source images."""
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COLORS = {"c2c": (255, 140, 0), "frozen": (20, 150, 255), "finetuned": (40, 210, 90)}


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_predictions(path):
    return {(r["sample_id"], r["concept"]): r for r in read_rows(path)}


def draw_panel(canvas, x, y, width, image_path, box, color, title):
    draw = ImageDraw.Draw(canvas)
    draw.text((x + 3, y + 2), title, fill=(0, 0, 0))
    image = Image.open(image_path).convert("RGB").resize((width, width))
    canvas.paste(image, (x, y + 22))
    if box:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle((x + x1*width, y + 22 + y1*width,
                        x + x2*width, y + 22 + y2*width), outline=color, width=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", type=Path, default=HERE / "runs/c2c_box_sae_003/test_predictions.csv")
    parser.add_argument("--frozen", type=Path, default=HERE / "runs/frozen_sae_box_baseline_002/test_predictions.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "backprop/manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "visualizations")
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()

    manifest = {r["sample_id"]: r for r in read_rows(args.manifest)}
    fine, frozen = load_predictions(args.fine), load_predictions(args.frozen)
    args.output.mkdir(parents=True, exist_ok=True)
    tile, row_height, label_height = args.tile, args.tile + 26, 28
    for concept in ("kimono", "flower"):
        positives, negatives = [], []
        for (sample_id, c), row in fine.items():
            if c != concept:
                continue
            if row["c2c_box_xyxy_normalized"]:
                positives.append((sample_id, row))
            else:
                negatives.append((sample_id, row))
        positives.sort(key=lambda item: float(item[1]["iou"]))
        chosen = []
        if positives:
            for i in sorted({0, len(positives)//2, len(positives)-1}):
                chosen.append((positives[i][0], positives[i][1], "C2C positive"))
        if negatives:
            fp = max(negatives, key=lambda item: float(item[1]["score"]))
            chosen.append((fp[0], fp[1], "highest-score C2C negative"))

        sheet = Image.new("RGB", (4*tile, len(chosen)*row_height + len(chosen)*label_height + 8), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 3), f"{concept.title()} held-out examples | C2C orange | frozen SAE blue | fine-tuned SAE green", fill="black")
        for i, (sample_id, fine_row, kind) in enumerate(chosen):
            y = 24 + i*(row_height + label_height)
            gt = json.loads(fine_row["c2c_box_xyxy_normalized"]) if fine_row["c2c_box_xyxy_normalized"] else None
            base = frozen[(sample_id, concept)]
            ft_box = json.loads(fine_row["predicted_box_xyxy_normalized"])
            base_box = json.loads(base["predicted_box_xyxy_normalized"])
            source = ROOT / "backprop" / manifest[sample_id]["image"]
            draw.text((4, y), f"{sample_id} — {kind} | frozen score {float(base['score']):.3f}, fine-tuned {float(fine_row['score']):.3f}, fine-tuned IoU {fine_row['iou'] or 'n/a'}", fill="black")
            panel_y = y + label_height
            draw_panel(sheet, 0*tile, panel_y, tile, source, None, None, "Original")
            draw_panel(sheet, 1*tile, panel_y, tile, source, gt, COLORS["c2c"], "C2C target")
            draw_panel(sheet, 2*tile, panel_y, tile, source, base_box, COLORS["frozen"], "Frozen SAE")
            draw_panel(sheet, 3*tile, panel_y, tile, source, ft_box, COLORS["finetuned"], "Fine-tuned SAE")
        sheet.save(args.output / f"{concept}_examples.jpg", quality=92)

        # All C2C-positive held-out examples, split into readable pages.
        page_size = 6
        for page, start in enumerate(range(0, len(positives), page_size), 1):
            subset = positives[start:start+page_size]
            full = Image.new("RGB", (4*tile, len(subset)*(row_height+label_height)+8), "white")
            d = ImageDraw.Draw(full)
            d.text((5, 3), f"{concept.title()} C2C-positive held-out set, page {page}", fill="black")
            for i, (sample_id, fine_row) in enumerate(subset):
                y = 24 + i*(row_height+label_height)
                gt = json.loads(fine_row["c2c_box_xyxy_normalized"])
                base = frozen[(sample_id, concept)]
                base_box = json.loads(base["predicted_box_xyxy_normalized"])
                ft_box = json.loads(fine_row["predicted_box_xyxy_normalized"])
                source = ROOT / "backprop" / manifest[sample_id]["image"]
                d.text((4, y), f"{sample_id} | C2C orange, frozen blue, fine-tuned green | IoU {fine_row['iou']}", fill="black")
                panel_y = y + label_height
                draw_panel(full, 0*tile, panel_y, tile, source, None, None, "Original")
                draw_panel(full, 1*tile, panel_y, tile, source, gt, COLORS["c2c"], "C2C target")
                draw_panel(full, 2*tile, panel_y, tile, source, base_box, COLORS["frozen"], "Frozen SAE")
                draw_panel(full, 3*tile, panel_y, tile, source, ft_box, COLORS["finetuned"], "Fine-tuned SAE")
            full.save(args.output / f"{concept}_positive_page_{page}.jpg", quality=92)
        print(f"{concept}: {len(positives)} positive, {len(negatives)} negative; wrote selected examples and all positive pages")


if __name__ == "__main__":
    main()
