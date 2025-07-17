# coding: utf-8

import argparse
import os
import pprint
from pathlib import Path

from matplotlib import pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, ConvNextFeatureExtractor

from orientation_angle_and_flip_detection.model import ImageOrientationAngleAndFlipOutputWithNoAttention, OAaFDNet

supported_extensions = {ex for ex, f in Image.registered_extensions().items() if f in Image.OPEN}


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset from folder")
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True, help="Model path")
    parser.add_argument("-i", "--input_paths", type=str, nargs="+", default=[], help="Input paths")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def predict_one(
    image_path: os.PathLike,
    model: OAaFDNet,
    processor: ConvNextFeatureExtractor,
) -> Image.Image:
    if Path(image_path).suffix not in supported_extensions:
        raise TypeError("Not support image extension")

    image = Image.open(image_path)
    image = image.convert("RGB")
    output: ImageOrientationAngleAndFlipOutputWithNoAttention = model(**processor(image, return_tensors="pt"))

    logits_angle = output.logits_angle
    logits_flip = output.logits_flip

    rotate_counts = list(logits_angle).index(1) if sum(logits_angle) else 0
    need_flip = True if logits_flip[1] == 1 else False

    for _ in range(rotate_counts):
        image = image.rotate(90, expand=True)
    image = image.transpose(Image.FLIP_LEFT_RIGHT) if need_flip else image

    return image


def predict(
    model_name_or_path: str,
    input_paths: list[os.PathLike],
    tqdm: bool = True,
):
    processor: ConvNextFeatureExtractor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model = OAaFDNet.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
    )

    for input_path in input_paths:
        if Path(input_path).is_file():
            image = predict_one(
                image_path=input_path,
                model=model,
                processor=processor,
            )
        else:
            for category_folder in Path(input_path).iterdir():
                for image_path in category_folder.iterdir():
                    image = predict_one(
                        image_path=image_path,
                        model=model,
                        processor=processor,
                    )
                    plt.show(image)


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    predict(**args_dict)
