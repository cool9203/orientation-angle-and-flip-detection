# coding: utf-8

import argparse
import ast
import os
import pprint
from pathlib import Path


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis data")
    parser.add_argument("-i", "--input_path", type=str, help="Input paths")

    args = parser.parse_args()

    return args


def analysis_data(
    input_path: os.PathLike,
):
    # Load label data
    labels: dict[str, dict[str, str]] = dict()
    with Path(input_path).open(mode="r", encoding="utf-8") as f:
        if Path(input_path).suffix in [".json"]:
            _iter_data = ast.literal_eval(f.read())
        elif Path(input_path).suffix in [".jsonl"]:
            _iter_data = list(ast.literal_eval(line) for line in f.readlines())
        else:
            raise TypeError("Not support")

        for info in _iter_data:
            category = info["category"]
            image_path = info["image_path"]
            angle = info["angle"]
            flip = info["flip"]
            if category not in labels:
                labels[category] = dict()
            labels[category][Path(image_path).name] = dict(
                angle=str(angle),
                flip=str(flip).lower(),
            )

    # Analysis

    for category, image_info in labels.items():
        all_angles = list()
        all_flips = list()
        for image_name, info in image_info.items():
            all_angles.append(info["angle"])
            all_flips.append(info["flip"])

        print(f"{category} Angles:")
        print(f"  - 0: {all_angles.count('0')}, {all_angles.count('0') / len(all_angles)}")
        print(f"  - 90: {all_angles.count('90')}, {all_angles.count('90') / len(all_angles)}")
        print(f"  - 180: {all_angles.count('180')}, {all_angles.count('180') / len(all_angles)}")
        print(f"  - 270: {all_angles.count('270')}, {all_angles.count('270') / len(all_angles)}")
        print(f"{category} Flip:")
        print(f"  - false: {all_flips.count('false')}, {all_flips.count('false') / len(all_flips)}")
        print(f"  - true: {all_flips.count('true')}, {all_flips.count('true') / len(all_flips)}")


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    analysis_data(**args_dict)
