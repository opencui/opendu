from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from core.commons import SimplePrompt
from builders.viggo import Viggo


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

    def __call__(self, utterance, prompt):
        formatted_prompt = prompt({"utterance": utterance})
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
    return x.split("(")[0]


if __name__ == "__main__":
    fprompt = SimplePrompt("<s> Convert the input text to structured representation. ### Input: {{utterance}} ### Output:")
    iprompt = SimplePrompt("<s> What is the meaning of the following input text, pick one from [ 'inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation','recommend', 'request_attribute' ] Input: {{utterance}} Output:")

    convert = Converter("./output/503B_FT_lr1e-5_ep5_top1_2023-09-26/checkpoint-800/")
    utterance = "Are you into third person PC games like Little Big Adventure?"
    utterance = "One game I got into recently on my Xbox is FIFA 12. I mean, on the whole it's pretty much par for the course for sports games, and EA Canada made it, so again, par for the course."

    dataset = Viggo("full").build("test")
    counts = [0, 0]
    for item in dataset:
        sequences = convert(item["utterance"], fprompt)
        counts[0] += 1
        seq = sequences[0]
        text = seq['generated_text']
        idx = text.index("Output:")
        result = text[idx+7:].strip()
        item_id = item["id"]
        print(f"Id:{item_id}\n")
        result = get_func(result)
        target = get_func(item['output'])
        if result == target:
            counts[1] += 1
        else:
            print(f"{result} != {target}\n")
    print(counts)