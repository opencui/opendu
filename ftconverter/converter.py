from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from builders.commons import SimplePrompt


#
# Converter is a lower level component of structifier. This directly use the model.
#
class Converter:
    def __init__(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            tokenizer=tokenizer,
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def __call__(self, utterance, prompt):
        formatted_prompt = prompt({"utterance": utterance})
        sequences = self.pipeline(
            formatted_prompt,
            do_sample=True,
            top_k=8,
            top_p = 0.9,
            num_return_sequences=1,
            repetition_penalty=1.1,
            max_new_tokens=1024,
        )
        return sequences


if __name__ == "__main__":
    prompt = SimplePrompt("Convert the input to structured representation. Input: {{utterance}} Output:")
    convert = Converter("./output/503B_FT_lr1e-5_ep5_top1_2023-08-25/checkpoint-6380/")
    utterance = "Are you into third person PC games like Little Big Adventure?"
    sequences = convert(utterance, prompt)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}\n")