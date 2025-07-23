# coding: utf-8

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from transformers import AutoImageProcessor, ConvNextFeatureExtractor, Trainer, TrainingArguments
from trl import ModelConfig, ScriptArguments, TrlParser

from orientation_angle_and_flip_detection.model import OAaFDNet

CLASS_NAME = "{flip}-{angle}"
IMAGE_CACHE_NAME = "{category}-{flip}-{angle}"


def generate_balance_dataset(
    all_data: dict[str, dict[str, list[dict[str, str | int | float | Image.Image]]]],
    image_1_cache: dict[str, Image.Image],
    image_count: int = None,
    seed: int | float | str | bytes | bytearray | None = None,
) -> tuple[
    dict[str, dict[str, list[dict[str, str | int | float | Image.Image]]]],
    dict[str, Image.Image],
]:
    print("Auto generate balance dataset")
    print(f"image_count: {image_count}")

    image_count = (
        image_count
        if image_count
        else max([len(payloads) for category, class_data in all_data.items() for class_name, payloads in class_data.items()])
    )

    complete_data = dict()
    for category, class_data in all_data.items():
        for class_name, payloads in class_data.items():
            if category not in complete_data:
                complete_data[category] = dict()
            complete_data[category][class_name] = list()

    rng = random.Random(seed)
    for category, class_data in all_data.items():
        for class_name, payloads in class_data.items():
            for index in range(image_count - len(payloads)):
                random_index = None
                while random_index is None:
                    class_name_random_index = rng.randint(0, len(class_data.keys()) - 1)
                    class_name_random = list(class_data.keys())[class_name_random_index]
                    if len(all_data[category][class_name_random]) > 0 and class_name_random != class_name:
                        random_index = rng.randint(0, len(all_data[category][class_name_random]) - 1)
                target_angle = int(class_name.split("-")[1])
                target_flip = class_name.split("-")[0].lower() == "true"

                source_angle = all_data[category][class_name_random][random_index]["angle"]
                source_flip = all_data[category][class_name_random][random_index]["flip"]
                base_image_2 = all_data[category][class_name_random][random_index]["image_2"].copy()
                source_image_2 = (base_image_2.transpose(Image.FLIP_LEFT_RIGHT) if source_flip else base_image_2).rotate(
                    source_angle, expand=True
                )  # Restore to non-flip and 0-angle

                random_angle = rng.randint(0, 3) * 90
                random_flip = rng.randint(0, 1) == 1
                image_1_name_new = IMAGE_CACHE_NAME.format(category=category, flip=random_angle, angle=random_flip)
                if image_1_name_new not in image_1_cache:
                    base_image_1_name = IMAGE_CACHE_NAME.format(category=category, flip=False, angle=0)
                    base_image_1 = image_1_cache.get(base_image_1_name).copy()
                    image_1_new = base_image_1.rotate(random_angle, expand=True)
                    image_1_new = image_1_new.transpose(Image.FLIP_LEFT_RIGHT) if random_flip else image_1_new
                    image_1_cache[image_1_name_new] = image_1_new

                # To image_1_new
                image_2_new = source_image_2.rotate(random_angle, expand=True)
                image_2_new = image_2_new.transpose(Image.FLIP_LEFT_RIGHT) if random_flip else image_2_new
                # To target flip and angle
                image_2_new = image_2_new.rotate(target_angle * (-1), expand=True) if target_angle != 0 else image_2_new
                image_2_new = image_2_new.transpose(Image.FLIP_LEFT_RIGHT) if target_flip else image_2_new

                complete_data[category][class_name].append(
                    dict(
                        image_1_name=image_1_name_new,
                        image_2=image_2_new,
                        angle=target_angle,
                        flip=target_flip,
                    )
                )

    for category, class_data in all_data.items():
        for class_name, _ in class_data.items():
            all_data[category][class_name] += complete_data[category][class_name]

    return (all_data, image_1_cache)


def preprocess_dataset(
    dataset_path: os.PathLike,
    image_count: int = None,
    balance: bool = False,
    seed: int | float | str | bytes | bytearray | None = None,
):
    image_1_cache: dict[str, Image.Image] = dict()
    all_data: dict[str, dict[str, list[dict[str, str | int | float | Image.Image]]]] = dict()

    # Pre-load all data
    with Path(dataset_path).open(mode="r", encoding="utf-8") as f:
        for payload in tqdm.tqdm(list(json.load(fp=f))):
            angle = payload.get("angle", None) or payload.get("label_angle", None)
            flip = payload.get("flip", None) or payload.get("label_flip", None)
            angle = int(angle) % 360
            flip = str(flip).lower() in ["1", "true", "yes"]
            category = Path(payload["image_1"]).stem
            class_name = CLASS_NAME.format(flip=flip, angle=angle)
            image_1_name = IMAGE_CACHE_NAME.format(category=category, flip=False, angle=0)

            if category not in all_data:
                all_data[category] = dict()
                for _flip in [False, True]:
                    for _angle in [0, 90, 180, 270]:
                        class_name = CLASS_NAME.format(flip=_flip, angle=_angle)
                        if class_name not in all_data[category]:
                            all_data[category][class_name] = list()

            if category not in image_1_cache:
                image_1_cache[image_1_name] = Image.open(payload["image_1"]).convert("RGB")

            all_data[category][class_name].append(
                {
                    "image_1_name": image_1_name,
                    "image_2": Image.open(payload["image_2"]).convert("RGB"),
                    "labels": [[angle], [flip]],
                    "angle": angle,
                    "flip": flip,
                }
            )

    (all_data, image_1_cache) = (
        generate_balance_dataset(
            all_data=all_data,
            image_1_cache=image_1_cache,
            image_count=image_count,
            seed=seed,
        )
        if balance
        else (all_data, image_1_cache)
    )

    dataset = list()
    for category, class_data in all_data.items():
        for class_name, payloads in class_data.items():
            for payload in payloads:
                dataset.append(
                    {
                        "pixel_values_1": image_1_cache.get(payload["image_1_name"]),
                        "pixel_values_2": payload["image_2"],
                        "labels": [
                            [int(payload["angle"]) // 90],
                            [int(payload["flip"])],
                        ],
                    }
                )
    return dataset


def compute_metrics(eval_pred):
    predictions_angle = np.argmax(eval_pred.predictions[0], axis=1)
    predictions_flip = np.argmax(eval_pred.predictions[1], axis=1)
    correct = 0
    for index in range(len(predictions_angle)):
        if (
            predictions_angle[index] == eval_pred.label_ids[0][index][0]
            and predictions_flip[index] == eval_pred.label_ids[1][index][0]
        ):
            correct += 1
    return {"accuracy": correct / len(predictions_angle)}


def train(script_args, training_args, model_args):
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    processor: ConvNextFeatureExtractor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    model = OAaFDNet.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels_angle=4,
        num_labels_flip=2,
        **model_kwargs,
    )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        pixel_values_1 = processor([example["pixel_values_1"] for example in examples], return_tensors="pt")
        pixel_values_2 = processor([example["pixel_values_2"] for example in examples], return_tensors="pt")
        labels = [
            torch.LongTensor([example["labels"][0] for example in examples]),
            torch.LongTensor([example["labels"][1] for example in examples]),
        ]
        return dict(
            pixel_values_1=pixel_values_1["pixel_values"],
            pixel_values_2=pixel_values_2["pixel_values"],
            labels=labels,
        )

    ################
    # Dataset
    ################
    dataset_names = str(script_args.dataset_name).split(",")

    if np.array([True if Path(dataset_path, "train.json").exists() else False for dataset_path in dataset_names]).all():
        train_data = list()
        val_data = list()
        for dataset_path in dataset_names:
            train_data += preprocess_dataset(dataset_path=Path(dataset_path, "train.json"), balance=True, seed=training_args.seed)
            if Path(dataset_path, "val.json").exists():
                val_data += preprocess_dataset(dataset_path=Path(dataset_path, "val.json"), balance=False)

        print(f"All train data: {len(train_data)}")
        print(f"All val data: {len(val_data)}")

        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "val": Dataset.from_list(val_data),
            }
            if val_data
            else {
                "train": Dataset.from_list(train_data),
            }
        )
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    train(script_args, training_args, model_args)
