# coding: utf-8

import argparse
import ast
import json
import os
import pprint
from pathlib import Path

import tqdm as TQDM


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset from folder")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input path")
    parser.add_argument("-ir", "--input_reference_image_path", type=str, required=True, help="Input reference image path")
    parser.add_argument("-l", "--label_path", type=str, required=True, help="Input label data path")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Output path")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def from_folder(
    input_path: os.PathLike,
    input_reference_image_path: os.PathLike,
    label_path: os.PathLike,
    output_path: os.PathLike = None,
    tqdm: bool = True,
) -> dict[str, list[dict[str, str | Path]]]:
    dataset: dict[str, list[dict[str, str | Path]]] = dict()
    dataset_info: dict[str, dict[str, dict[str, str | int | float]]] = dict()
    reference_images: dict[str, str | Path] = dict()

    # Load reference image data
    for reference_image_path in (
        TQDM.tqdm(list(Path(input_reference_image_path).iterdir()), desc="Load reference images")
        if tqdm
        else Path(input_reference_image_path).iterdir()
    ):
        reference_images[reference_image_path.stem] = reference_image_path

    # Load label data
    with Path(label_path).open(mode="r", encoding="utf-8") as f:
        if Path(label_path).suffix in [".json"]:
            _iter_data = ast.literal_eval(f.read())
        elif Path(label_path).suffix in [".jsonl"]:
            _iter_data = list(ast.literal_eval(line) for line in f.readlines())
        else:
            raise TypeError("Not support")

        for info in TQDM.tqdm(_iter_data, desc="Process label data") if tqdm else _iter_data:
            category = info["category"]
            image_path = info["image_path"]
            angle = info["angle"]
            flip = info["flip"]
            if category not in dataset_info:
                dataset_info[category] = dict()
            dataset_info[category][Path(image_path).name] = dict(
                angle=str(angle),
                flip=str(flip),
            )

    # Generate data
    for category_folder in (
        TQDM.tqdm(list(Path(input_path).iterdir()), desc="Generate train data") if tqdm else Path(input_path).iterdir()
    ):
        if category_folder.is_file():
            continue
        dataset[category_folder.name] = list()
        for image_path in (
            TQDM.tqdm(list(category_folder.iterdir()), desc=category_folder.name, leave=False)
            if tqdm
            else category_folder.iterdir()
        ):
            dataset[category_folder.name].append(
                {
                    "image_1": str(reference_images[category_folder.name]),
                    "image_2": str(image_path),
                    "label_angle": str(dataset_info[category_folder.name][image_path.name]["angle"]),
                    "label_flip": str(dataset_info[category_folder.name][image_path.name]["flip"]),
                }
            )

    if output_path:
        save_path = Path(Path(output_path).parent, f"{Path(output_path).stem}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open(mode="w", encoding="utf-8") as f:
            json.dump(obj=list(item for v in dataset.values() for item in v), fp=f, ensure_ascii=False)

    return dataset


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    from_folder(**args_dict)
