"""
get_descriptions.py 调用第三方语言模型生成类别描述

    @Time    : 2025/05/12
    @Author  : JackWang
    @File    : get_description.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import json
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal, Optional

# Third-Party Library
from openai import OpenAI
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion

# Torch Library

# My Library


@dataclass
class ClassNames:

    voc: list[str] = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    coco: list[str] = (
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )


# Sec.3.2. Text Semantic Enrichment
instruction = """ List {n} descriptions with key properties to describe the [CLASS] in terms of appearance, color, shape, size, or material, etc. These descriptions will help visually distinguish the [CLASS] from other classes in the dataset.  Each description should follow the format: 'a clean [CLASS]. it + descriptive contexts.'. You give {n} descriptions directly. Do not add any other information. The descriptions should be in English. The descriptions should be unique and not repeated. The descriptions should be relevant to the [CLASS] and should not include any irrelevant information.
"""


def _get_client(
    which: Literal["gpt4", "deepseek"],
) -> tuple[OpenAI, Callable[[str], ChatCompletion]]:
    apikey = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parent / "apikeys.yaml")
    )["apikeys"][which]

    client = OpenAI(api_key=apikey["key"], base_url=apikey["base_url"])

    def func(prompt: str) -> str:
        return client.chat.completions.create(
            model="gpt-4" if which == "gpt4" else "deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates detailed visual descriptions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

    return (client, func)


def _get_description_generator(
    which: Literal["gpt4", "deepseek"],
) -> Callable[[str, int], list[str]]:

    _, func = _get_client(which)

    def get_descriptions(class_name: str, n: int) -> list[str]:
        prompt = instruction.replace("{n}", str(n)).replace("[CLASS]", class_name)
        response = func(prompt)
        content = response.choices[0].message.content
        return [i.strip() for i in content.split("\n") if i.strip()]

    return get_descriptions


def get_descriptions(
    dataset: Literal["voc", "coco"],
    class_names: list[str],
    n: int,
    which: Optional[Literal["gpt4", "deepseek"]] = "deepseek",
) -> dict[str, list[str]]:

    if (
        description_file := Path(__file__).resolve().parent / "descriptions.json"
    ).exists():
        with open(description_file, "r") as f:
            cached_descriptions: dict[str, dict[str, list[str]]] = json.load(f)

        if dataset in cached_descriptions:
            cache = cached_descriptions[dataset]
            for class_name in class_names:
                assert (
                    len(cache[class_name]) == n
                ), f"Cache file found, but the number of descriptions for {class_name} is less than {n}. Remove the cache file to regenerate."
            return cache
    else:
        cached_descriptions = {}
        assert (
            which is not None
        ), "Cache file not found, please specify which model to use to generate descriptions."

    get_descriptions = _get_description_generator(which)
    descriptions = {
        class_name: get_descriptions(class_name, n) for class_name in class_names
    }
    cached_descriptions |= {dataset: descriptions}
    with open(description_file, "w") as f:
        json.dump(cached_descriptions, f, indent=4)

    return descriptions


if __name__ == "__main__":
    import torch
    from . import clip

    descriptions = get_descriptions("voc", ClassNames.voc, 20, "deepseek")
    descriptions = sum(descriptions.values(), [])

    device = torch.device("mps")
    text_tokens = clip.tokenize(descriptions, truncate=True).to(device)
    print(text_tokens.shape)
