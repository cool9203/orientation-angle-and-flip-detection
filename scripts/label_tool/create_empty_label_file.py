# coding: utf-8

import argparse
import json
import os
import pprint
from pathlib import Path

import tqdm as TQDM
from PIL import Image

supported_extensions = {ex for ex, f in Image.registered_extensions().items() if f in Image.OPEN}


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset from folder")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input path")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path")
    parser.add_argument("-f", "--output_format", type=str, choices=["json", "jsonl"], default="json", help="Output format")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def create_empty_label_file(
    input_path: os.PathLike,
    output_path: os.PathLike = None,
    output_format: os.PathLike = None,
    tqdm: bool = True,
) -> list[dict[str, str | int | float]]:
    output_format = (
        output_format
        if output_format
        else (Path(output_path).suffix[1:] if output_path and Path(output_path).suffix else output_format)
    )

    labels: list[dict[str, str | int | float]] = list()
    for category_folder in TQDM.tqdm(list(Path(input_path).iterdir())) if tqdm else Path(input_path).iterdir():
        if category_folder.is_file():
            continue
        for image_path in (
            TQDM.tqdm(list(category_folder.iterdir()), desc=category_folder.name, leave=False)
            if tqdm
            else category_folder.iterdir()
        ):
            if image_path.suffix not in supported_extensions:
                continue
            labels.append(
                dict(
                    category=category_folder.name,
                    image_path=str(image_path.resolve()),
                    angle="0",
                    flip="true",
                )
            )

    if output_path:
        save_path = Path(Path(output_path).parent, f"{Path(output_path).stem}.{output_format}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open(mode="w", encoding="utf-8") as f:
            if output_format in ["json"]:
                json.dump(obj=labels, fp=f, indent=4, ensure_ascii=False)
            elif output_format in ["jsonl"]:
                for label in labels:
                    f.write(json.dumps(obj=label, ensure_ascii=False) + "\n")
            else:
                raise ValueError(f"Not support output format: '{output_format}'")

    return labels


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    create_empty_label_file(**args_dict)
