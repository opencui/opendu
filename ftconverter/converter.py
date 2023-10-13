from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from builders.commons import SimplePrompt


#
# Converter is a lower level component of structifier. This directly use the model.
#
class Converter:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto")

        self.pipeline = transformers.pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=model
        )

    def __call__(self, utterance, prompt):
        formatted_prompt = prompt({"utterance": utterance})
        return self.pipeline(
            f"{formatted_prompt}{self.tokenizer.bos_token}",
            do_sample=True,
            top_k=8,
            top_p = 0.9,
            num_return_sequences=1,
            repetition_penalty=1.1,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    prompt = SimplePrompt("Convert the input text to structured representation. ### Input: {{utterance}} ### Output:")
    prompt = SimplePrompt("What is the meaning of the following input text, pick one from [ 'inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation','recommend', 'request_attribute' ] Input: {{utterance}} Output:")

    convert = Converter("./output/503B_FT_lr1e-5_ep5_top1_2023-09-25/checkpoint-800/")
    utterance = "Are you into third person PC games like Little Big Adventure?"
    utterance = "One game I got into recently on my Xbox is FIFA 12. I mean, on the whole it's pretty much par for the course for sports games, and EA Canada made it, so again, par for the course."
    sequences = convert(utterance, prompt)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}\n\n")