import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from builders.viggo import Viggo
from core.prompt import get_prompt


#
# Converter is a lower level component of structifier. This directly use the model.
#
class Converter:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto")

        self.pipeline = transformers.pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=model
        )

    def __call__(self, item, prompt):
        formatted_prompt = prompt(item)
        return self.pipeline(
            formatted_prompt,
            do_sample=True,
            top_k=8,
            top_p = 0.9,
            num_return_sequences=1,
            repetition_penalty=1.1,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
        )


def get_func(x):
    #return x.split("(")[0]
    return x


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    viggo = Viggo()
    output = "./index/viggo/"
    prompt = get_prompt(viggo, output)

    convert = Converter("./output/503B_FT_lr1e-5_ep5_top1_2023-09-26/checkpoint-3190/")

    dataset = Viggo("full").build("test")
    counts = [0, 0]
    marker = "### Output:"
    for item in dataset:
        sequences = convert(item, prompt)
        counts[0] += 1
        seq = sequences[0]
        text = seq['generated_text']
        idx = text.index(marker)
        result = text[idx+len(marker):].strip()
        item_id = item["id"]
        result = get_func(result)
        target = get_func(item['target_full'])
        if result == target:
            counts[1] += 1
        else:
            print(f"{result} != {target}\n")
    print(counts)