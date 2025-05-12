"""
get_descriptions.py 调用第三方语言模型生成类别描述

    @Time    : 2025/05/12
    @Author  : JackWang
    @File    : get_description.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from pathlib import Path
from typing import Literal
from functools import partial
from collections.abc import Callable

# Third-Party Library
from openai import OpenAI
from omegaconf import DictConfig, OmegaConf
from openai.types.chat import ChatCompletion

# Torch Library

# My Library
from .classenames import ClassNames


# Sec.3.2. Text Semantic Enrichment
instruction = """ List {n} descriptions with key properties to describe the [CLASS] in terms of appearance, color, shape, size, or material, etc. These descriptions will help visually distinguish the [CLASS] from other classes in the dataset.  Each description should follow the format: 'a clean [CLASS]. it + descriptive contexts.'. You give {n} descriptions directly. Do not add any other information. The descriptions should be in English. The descriptions should be unique and not repeated. The descriptions should be relevant to the [CLASS] and should not include any irrelevant information.
"""


def get_client(
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


def get_description_generator(
    which: Literal["gpt4", "deepseek"],
) -> Callable[[str, int], list[str]]:

    _, func = get_client(which)

    def get_descriptions(class_name: str, n: int) -> list[str]:
        prompt = instruction.replace("{n}", str(n)).replace("[CLASS]", class_name)
        response = func(prompt)
        content = response.choices[0].message.content
        return [i.strip() for i in content.split("\n") if i.strip()]

    return get_descriptions


def get


if __name__ == "__main__":

    get_descriptions = get_description_generator("deepseek")

    descriptions = get_descriptions(
        class_name="cat",
        n=20,
    )

    print(descriptions)
