"""Nepali image captioning with your fine-tuned BLIP model.

Usage:
    python caption.py --model path/to/best_model photo1.jpg photo2.png some_folder/
Results are printed and also saved to captions.csv (opens in Excel with Nepali text intact).
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    BertTokenizerFast,  # type: ignore[reportAttributeAccessIssue]
    BlipForConditionalGeneration,
    BlipProcessor,
)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")  # so Nepali prints correctly on Windows


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Skipping (not found): {p}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="folder containing model.safetensors, tokenizer.json, ...",
    )
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--out", default="captions.csv")
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = BlipProcessor.from_pretrained(args.model)
    # strip_accents=False keeps Devanagari vowel signs; must match training
    processor.tokenizer = BertTokenizerFast.from_pretrained(
        args.model, strip_accents=False
    )
    model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(
        args.model
    )
    model = model.to(device)  # type: ignore[reportArgumentType]
    model.eval()

    check = processor.tokenizer.tokenize("एउटा कुकुर दौडिरहेको छ")
    print("TEST: Tokenizer check (expect ['एउटा', 'कुकुर', 'दौडिरहेको', 'छ']):", check)

    files = collect_images(args.images)
    if not files:
        sys.exit("No images found.")

    rows = []
    with torch.no_grad():
        for f in files:
            image = Image.open(f).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)  # pyright: ignore[reportCallIssue]
            ids = model.generate(
                **inputs,
                num_beams=args.beams,
                max_length=40,
                min_length=3,
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True,
                do_sample=True,
            )
            caption = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            print(f"{f.name}: {caption}")
            rows.append((f.name, caption))

    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "caption"])
        w.writerows(rows)
    print("Saved:", args.out)

    txt_out = Path(args.out).with_suffix(".txt")
    with open(txt_out, "w", encoding="utf-8") as fh:
        for name, caption in rows:
            fh.write(f"{name}: {caption}\n")
    print("Saved:", txt_out)


if __name__ == "__main__":
    main()
