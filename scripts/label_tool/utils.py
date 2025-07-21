# coding: utf-8

import ast
import os
from pathlib import Path


def load_labels(
    label_path: os.PathLike,
) -> dict[str, dict[str, dict[str, str | int | float]]]:
    # Load label data
    labels: dict[str, dict[str, dict[str, str | int | float]]] = dict()
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
            if category not in labels:
                labels[category] = dict()
            labels[category][Path(image_path).name] = dict(
                angle=str(angle),
                flip=str(flip).lower(),
            )
    return labels
