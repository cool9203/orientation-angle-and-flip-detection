# coding: utf-8

import argparse
import ast
import os
import pprint
from pathlib import Path

from matplotlib import pyplot as plt
from PIL import Image

supported_extensions = {ex for ex, f in Image.registered_extensions().items() if f in Image.OPEN}


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset from folder")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input path")
    parser.add_argument("-l", "--label_path", type=str, required=True, help="Input label data path")

    args = parser.parse_args()

    return args


def check_label_data(
    input_path: os.PathLike,
    label_path: os.PathLike,
) -> dict[str, list[dict[str, str | Path]]]:
    dataset_info: dict[str, dict[str, dict[str, str | int | float]]] = dict()

    # Load label data
    with Path(label_path).open(mode="r", encoding="utf-8") as f:
        if Path(label_path).suffix in [".json"]:
            _iter_data = ast.literal_eval(f.read())
        elif Path(label_path).suffix in [".jsonl"]:
            _iter_data = list(ast.literal_eval(line) for line in f.readlines())
        else:
            raise TypeError("Not support")

        for info in _iter_data:
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
    for category_folder in Path(input_path).iterdir():
        if category_folder.is_file():
            continue
        image_data = list()
        for image_path in category_folder.iterdir():
            if image_path.suffix not in supported_extensions:
                continue
            image = Image.open(image_path)

            # Flip
            image = (
                image.transpose(Image.FLIP_LEFT_RIGHT)
                if str(dataset_info[category_folder.name][image_path.name]["flip"]).lower() in ["true", "yes", "1"]
                else image
            )

            # Rotate
            for _ in range(int(dataset_info[category_folder.name][image_path.name]["angle"]) // 90):
                image = image.rotate(90, expand=True)

            image_data.append(
                {
                    "image": image,
                    "name": image_path.name,
                }
            )

            if len(image_data) == 10:
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                axes = axes.flatten()

                for i, data in enumerate(image_data):
                    axes[i].imshow(data["image"])
                    axes[i].set_title(data["name"])
                    axes[i].axis("off")
                plt.show()
                image_data.clear()
        if image_data:
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()

            for i, data in enumerate(image_data):
                axes[i].imshow(data["image"])
                axes[i].set_title(data["name"])
                axes[i].axis("off")
            plt.show()
            image_data.clear()


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    check_label_data(**args_dict)
