# coding: utf-8

import argparse
import os
import pprint
from pathlib import Path

import tqdm as TQDM
from matplotlib import pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, ConvNextFeatureExtractor

from orientation_angle_and_flip_detection.model import ImageOrientationAngleAndFlipOutputWithNoAttention, OAaFDNet

supported_extensions = {ex for ex, f in Image.registered_extensions().items() if f in Image.OPEN}


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset from folder")
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True, help="Model path")
    parser.add_argument("-i", "--input_paths", type=str, nargs="+", default=[], help="Input paths")
    parser.add_argument("-ir", "--input_reference_image_path", type=str, required=True, help="Input reference image path")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def predict_one(
    image_path_reference: os.PathLike,
    image_path_predict: os.PathLike,
    model: OAaFDNet,
    processor: ConvNextFeatureExtractor,
) -> Image.Image:
    image_reference = Image.open(image_path_reference)
    image_reference = image_reference.convert("RGB")
    image_predict = Image.open(image_path_predict)
    image_predict = image_predict.convert("RGB")

    output: ImageOrientationAngleAndFlipOutputWithNoAttention = model(
        pixel_values_1=processor(image_reference, return_tensors="pt")["pixel_values"],
        pixel_values_2=processor(image_predict, return_tensors="pt")["pixel_values"],
    )

    logits_angle = output.logits_angle
    logits_flip = output.logits_flip

    rotate_counts = logits_angle.argmax(dim=1)[0]
    need_flip = logits_flip.argmax(dim=1)[0] == 1

    image_predict = image_predict.transpose(Image.FLIP_LEFT_RIGHT) if need_flip else image_predict
    for _ in range(rotate_counts):
        image_predict = image_predict.rotate(90, expand=True)

    return image_predict


def predict(
    model_name_or_path: str,
    input_paths: list[os.PathLike],
    input_reference_image_path: os.PathLike,
    tqdm: bool = True,
):
    reference_images: dict[str, str | Path] = dict()

    # Load reference image data
    for reference_image_path in (
        TQDM.tqdm(list(Path(input_reference_image_path).iterdir()), desc="Load reference images")
        if tqdm
        else Path(input_reference_image_path).iterdir()
    ):
        if reference_image_path.suffix in supported_extensions:
            reference_images[reference_image_path.stem] = reference_image_path

    processor: ConvNextFeatureExtractor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model = OAaFDNet.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
    )

    for input_path in input_paths:
        if Path(input_path).is_file():
            raise TypeError("Not support pass file")

        for category_folder in Path(input_path).iterdir():
            images = list()
            if category_folder.is_file():
                continue
            for image_path in category_folder.iterdir():
                if image_path.suffix not in supported_extensions:
                    continue
                image = predict_one(
                    image_path_reference=reference_images[category_folder.name],
                    image_path_predict=image_path,
                    model=model,
                    processor=processor,
                )
                images.append(image)

                if len(images) == 10:
                    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                    axes = axes.flatten()

                    for i, img in enumerate(images):
                        axes[i].imshow(img)
                        axes[i].set_title(f"Image {i + 1}")
                        axes[i].axis("off")
                    plt.show()
                    images.clear()
            if images:
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                axes = axes.flatten()

                for i, img in enumerate(images):
                    axes[i].imshow(img)
                    axes[i].set_title(f"Image {i + 1}")
                    axes[i].axis("off")
                plt.show()
                images.clear()


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    predict(**args_dict)
