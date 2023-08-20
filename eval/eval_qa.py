import sys
import os

sys.path.append("..")

cuda_path = "/usr/local/cuda-11.7/bin/nvcc"
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] += f"{cuda_path}"
else:
    os.environ["LD_LIBRARY_PATH"] = cuda_path

import re
import json
import fire
import string

from tqdm.autonotebook import tqdm
from medalpaca.inferer import Inferer

greedy_search = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 128,
    "early_stopping": False
}

beam_search = {
    "num_beams": 4,
    "do_sample": False,
    "max_new_tokens": 128,
    "early_stopping": True,
}

sampling_top_k = {
    "do_sample": True,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.7,
    "top_k": 50
}

sampling_top_p = {
    "do_sample": True,
    "top_k": 0,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.7,
    "top_p": 0.9
}

sampling = {
    "do_sample": True,
    "top_k": 50,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.4,
    "top_p": 0.9
}


def format_question(d):
    question = d["question"]
    options = d["options"]
    for k, v in options.items():
        question += f"\n{k}: {v}"
    return question


def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str

    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[
        start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[
        end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""


def starts_with_capital_letter(input_str):
    """
    The answers should start like this: 
        'A: '
        'A. '
        'A '
    """
    pattern = r'^[A-Z](:|\.|) .+'
    return bool(re.match(pattern, input_str))


def main(
        model_name: str,  # "medalpaca/medalpaca-lora-13b-8bit",
        prompt_template: str,
        # "../medalpaca/prompt_templates/medalpaca.json",
        base_model: str,  # "decapoda-research/llama-13b-hf",
        peft: bool,  # True,
        load_in_8bit: bool,  # True
        path_to_exams: str,  # eval/data/test/
        ntries: int = 5,
        skip_if_exists: bool = True,
):
    model = Inferer(
        model_name=model_name,
        prompt_template=prompt_template,
        base_model=base_model,
        peft=peft,
        load_in_8bit=load_in_8bit,
    )

    question = "What is Georgia Tech known for?"

    response = model(
        instruction="Answer this multiple choice question.",
        input=format_question(question),
        output="The Answer to the question is:",
        **sampling
    )
    response = strip_special_chars(response)
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
