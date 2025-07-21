# coding: utf-8

import argparse
import os
import pprint

from utils import load_labels


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis data")
    parser.add_argument("-i", "--input_path", type=str, help="Input paths")

    args = parser.parse_args()

    return args


def analysis_data(
    input_path: os.PathLike,
) -> dict[str, dict[str, list[str]]]:
    # Load label data
    labels = load_labels(label_path=input_path)

    # Analysis
    all_data: dict[str, dict[str, list[str]]] = dict()
    for category, image_info in labels.items():
        data: dict[str, int] = dict()
        for image_name, info in image_info.items():
            name = f"{info['flip']} {info['angle']}"
            if name not in data:
                data[name] = list()
            data[name].append(image_name)

        all_data[category] = data  # Update to all data

    # Calc all distributed and save to all_data
    _all_data: dict[str, int] = dict()
    for category, data in all_data.items():
        for name, filenames in data.items():
            if name not in _all_data:
                _all_data[name] = list()
            _all_data[name] += filenames
    all_data["all"] = _all_data

    return all_data


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    pprint.pprint(args_dict)
    analysis_results = analysis_data(**args_dict)

    # Show distributed
    for category, data in analysis_results.items():
        names = sorted(data.keys())
        data_count = sum([len(filenames) for _, filenames in data.items()])

        print(f"{category}:")
        for name in names:
            print(f"  - {name}: {len(data[name])}, {len(data[name]) / data_count}")
